import argparse
import os

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qa_metrics.cfm import CFMatcher
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from transformers.generation import GenerationConfig

from models.salmonn import SALMONN
from models.via import VIA

torch.manual_seed(1234)
cfm = CFMatcher()

dials = [
    "aus",
    "gbr",
    "ind_n",
    "ind_s",
    "irl",
    "kenya",
    "nga",
    "nzl",
    "phl",
    "usa",
    "zaf",
]


@torch.no_grad
def get_response_pipeline(asr_model, model, audio, dial):
    value = audio[dial]
    text = asr_model(value)["text"]
    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Give answers as a simple single sentence.",
        },
        {"role": "user", "content": text},
    ]
    if "mistral" in model.name:
        chat = [
            {
                "role": "user",
                "content": "You are a helpful assistant. Give answers as a simple single sentence.\n\n"
                + text,
            },
        ]
    query = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")

    output = model.generate(query, max_new_tokens=128)
    # Mistral & Llama
    if "Llama-3" in model.name:
        split_token = "<|start_header_id|>assistant<|end_header_id|>"
        response = (
            tokenizer.decode(output[0], skip_special_tokens=False)
            .split(split_token)[-1]
            .strip()
            .replace("<|eot_id|>", "")
        )
    elif "qwen" not in model.name:
        split_token = "[/INST]"
        response = (
            tokenizer.decode(output[0], skip_special_tokens=True)
            .split(split_token)[-1]
            .strip()
        )
    else:
        split_token = "<|im_start|>"
        response = (
            tokenizer.decode(output[0], skip_special_tokens=False)
            .split(split_token)[-1]
            .strip()
            .replace("<|im_end|>", "")
        )

    return response


@torch.no_grad
def get_response_pipeline_qwen(asr_model, model, audio, dial):
    value = audio[dial]
    text = asr_model(value)["text"]

    query = tokenizer.from_list_format(
        [
            {"text": text + " Give a simple one sentence answer."},
        ]
    )

    response, history = model.chat(tokenizer, query=query, history=None)

    return response


@torch.no_grad
def get_response_end_to_end_s(model, audio, dial):
    value = audio[dial]
    sf.write("tmp.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            wav_path="tmp.wav",
            prompt="You are a helpful assistant. Give a simple one sentence answer.",
            do_sample=False,
            top_p=1.0,
        )
    response = llm_message[0]

    return response


@torch.no_grad
def get_response_end_to_end_v(model, audio, dial):
    value = audio[dial]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            audio=value["array"],
            prompt="You are a helpful assistant. Give a simple one sentence answer.",
        )
    response = llm_message

    return response


@torch.no_grad
def get_response_end_to_end_q(model, audio, dial):
    value = audio[dial]
    sf.write("tmp.wav", value["array"], value["sampling_rate"], format="wav")
    query = tokenizer.from_list_format(
        [{"audio": "tmp.wav"}, {"text": "Give a simple one sentence answer."}]
    )

    response, history = model.chat(
        tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
    )
    return response


parser = argparse.ArgumentParser("sdqa_args")
parser.add_argument("m_type", help="path for transcript file", type=str)
parser.add_argument("model_name", help="path for transcript file", type=str)
args = parser.parse_args()
m_type = args.m_type
model_name = args.model_name
# m_type = "pipe"
# m_type = "e2e"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "Qwen/Qwen1.5-7B-Chat"
# m_type = "e2e"
# model_name = "Qwen/Qwen-Audio-Chat"
# model_name = "salmonn_7b"
# model_name = "via"
if m_type == "e2e":
    if "Qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).eval()

        model.generation_config = GenerationConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            do_sample=False,
            top_k=50,
            top_p=1.0,
        )
    elif "salmonn" in model_name:
        model = SALMONN(
            ckpt="./SALMONN_PATHS/salmonn_v1.pth",
            whisper_path="./SALMONN_PATHS/whisper-large-v2",
            beats_path="./SALMONN_PATHS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            vicuna_path="./SALMONN_PATHS/vicuna-13b-v1.1",
            low_resource=False,
        )
    elif "via" in model_name:
        model = VIA("/data/wheld3/via-7b-3/model-00001-of-00004.safetensors")

else:
    asr_model_id = "openai/whisper-large-v3"

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        asr_model_id,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    processor = AutoProcessor.from_pretrained(asr_model_id)

    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    ).eval()
    model.name = model_name

    model.generation_config = GenerationConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        do_sample=False,
        max_new_tokens=1000,
        top_k=50,
        top_p=1.0,
        temperature=1.0,
    )

ds = load_dataset("WillHeld/SD-QA")["dev"].filter(lambda example: example["answers"])
dial_scores = {}
for dial in dials:
    scores = []
    name_short = model_name.lower().split("/")[-1]
    filename = f"./sdqa-res/{m_type}_{name_short}/{dial}_outs.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for idx, ex in enumerate(tqdm(ds)):
            try:
                id = ex["id"]
                if m_type == "e2e" and "qwen" in name_short:
                    pred = get_response_end_to_end_q(model, ex, dial)
                elif m_type == "e2e" and "salmonn" in name_short:
                    pred = get_response_end_to_end_s(model, ex, dial)
                elif m_type == "e2e" and "via" in name_short:
                    pred = get_response_end_to_end_v(model, ex, dial)
                elif m_type != "e2e" and (
                    "qwen" in name_short and "1.5" not in name_short
                ):
                    pred = get_response_pipeline_qwen(asr, model, ex, dial)
                else:
                    pred = get_response_pipeline(asr, model, ex, dial)
                scores = [
                    value[pred]
                    for value in cfm.get_scores(
                        ex["answers"], pred, ex["question"]
                    ).values()
                ]
                score = max(scores)
            except Exception as e:
                print(e)
                print(id)
                pred = "ERROR IN PROCESSING"
                score = 0.0
            scores.append(score)
            pred_rep = '"""' + pred.replace("\n", "[NEW_LINE]") + '"""'
            f.write(f"{id}[SEP_DIAL]{pred_rep}[SEP_DIAL]{score}\n")
            if idx % 10 == 0 and idx != 0:
                f.flush()
    dial_scores[dial] = np.mean(scores)
print(dial_scores)
