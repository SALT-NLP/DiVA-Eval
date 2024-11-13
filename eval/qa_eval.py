import argparse
import os

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qa_metrics.cfm import CFMatcher
from tqdm import tqdm
import librosa
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizer,
    pipeline,
    Qwen2AudioForConditionalGeneration,
    WhisperFeatureExtractor,
    AutoProcessor,
)
from blsp.blsp.src.modeling_blsp import BlspModel
from blsp.blsp.src.speech_text_paired_dataset import get_waveform
from transformers.generation import GenerationConfig

from load_via_eval import load_via_eval
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
    sf.write("tmp_s.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            wav_path="tmp_s.wav",
            prompt="You are a helpful assistant. Give a simple one sentence answer.",
            do_sample=False,
            top_p=1.0,
            max_new_tokens=64,
        )
    response = llm_message[0]

    return response


@torch.no_grad
def get_response_end_to_end_q2(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_q2.wav", value["array"], value["sampling_rate"], format="wav")
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "tmp_q.wav",
                },
                {
                    "type": "text",
                    "text": "Give a simple one sentence answer.",
                },
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = [
        librosa.load("tmp_q2.wav", sr=processor.feature_extractor.sampling_rate)[0]
    ]
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs["input_ids"] = inputs.input_ids.to("cuda:0")
    inputs["attention_mask"] = inputs.attention_mask.to("cuda:0")
    inputs["input_features"] = inputs.input_features.to("cuda:0")

    generate_ids = model.generate(**inputs, max_new_tokens=64)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


@torch.no_grad
def get_response_end_to_end_v(model, audio, dial):
    value = audio[dial]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            audio=value["array"],
            prompt="You are a helpful assistant. Give your answers as a simple single sentence.\n",
            max_new_tokens=64,
        )
    response = llm_message

    return response


@torch.no_grad
def get_response_blsp(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_b.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        instruction = "You are a helpful assistant. Give your answers as a simple single sentence."
        input_ids = tokenizer(
            f"###[Human]:{instruction}", return_tensors="pt"
        ).input_ids.cuda()
        audio = "tmp_b.wav"
        speech_values, speech_attention_mask = None, None
        speech = get_waveform(audio, output_sample_rate=extractor.sampling_rate)
        speech_inputs = extractor(
            speech,
            sampling_rate=extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        speech_values = speech_inputs.input_features.cuda()
        speech_attention_mask = speech_inputs.attention_mask.cuda()
        suffix_input_ids = (
            tokenizer("\n\n\n###[Assistant]:", return_tensors="pt")
            .input_ids[:, 1:]
            .cuda()
        )
        output = model.generate(
            input_ids=input_ids,
            suffix_input_ids=suffix_input_ids,
            speech_values=speech_values,
            speech_attention_mask=speech_attention_mask,
            generation_config=generation_config,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


@torch.no_grad
def get_response_end_to_end_q(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_q.wav", value["array"], value["sampling_rate"], format="wav")
    query = tokenizer.from_list_format(
        [{"audio": "tmp_q.wav"}, {"text": "Give a simple one sentence answer."}]
    )

    response, history = model.chat(
        tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
        max_new_tokens=64,
    )
    return response


parser = argparse.ArgumentParser("sdqa_args")
parser.add_argument("m_type", help="path for transcript file", type=str)
parser.add_argument("model_name", help="path for transcript file", type=str)
parser.add_argument(
    "--dataset_name",
    help="path for transcript file",
    type=str,
    default="Spoken_Dialect_QA",
)
args = parser.parse_args()
m_type = args.m_type
model_name = args.model_name
if m_type == "e2e":
    if "Qwen2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map="sequential"
        )

        model.generation_config = GenerationConfig.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True,
            do_sample=False,
            top_k=50,
            top_p=1.0,
        )
    elif "Qwen" in model_name:
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
    elif "blsp" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        model = BlspModel.from_pretrained(model_name).cuda()
        generation_config = GenerationConfig(
            max_new_tokens=128,
            min_new_tokens=1,
            do_sample=False,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            num_return_sequences=1,
        )
        generation_config.update(
            **{
                "max_new_tokens": 1,
                "min_new_tokens": 64,
                "do_sample": False,
                "top_p": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
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
        # model = VIA("../llama3-via-v0/model-00001-of-00004.safetensors")
        model = VIA("/data/wheld3/step-4299/model-00001-of-00004.safetensors")

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

dataset_name = args.dataset_name
dials = {
    "non_social_HeySquad_QA": ["default"],
    "Spoken_Dialect_QA": [
        "usa",
        "aus",
        "gbr",
        "ind_n",
        "ind_s",
        "irl",
        "kenya",
        "nga",
        "nzl",
        "phl",
        "zaf",
    ],
}[dataset_name]
dial_scores = {}
for dial in dials:
    x_label, y_label, ds = load_via_eval(dataset_name, language=dial)
    global_scores = []
    name_short = model_name.lower().split("/")[-1]
    filename = f"./{dataset_name}_tmp_Results/{m_type}_{name_short}/{dial}_outs.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for idx, ex in enumerate(tqdm(ds)):
            try:
                id = ex["id"]
                if m_type == "e2e" and "qwen2" in name_short:
                    pred = get_response_end_to_end_q2(model, ex, x_label)
                elif m_type == "e2e" and "qwen" in name_short:
                    pred = get_response_end_to_end_q(model, ex, x_label)
                elif m_type == "e2e" and "salmonn" in name_short:
                    pred = get_response_end_to_end_s(model, ex, x_label)
                elif m_type == "e2e" and "blsp" in name_short:
                    pred = get_response_blsp(model, ex, x_label)
                elif m_type == "e2e" and "via" in name_short:
                    pred = get_response_end_to_end_v(model, ex, x_label)
                elif m_type != "e2e" and (
                    "qwen" in name_short and "1.5" not in name_short
                ):
                    pred = get_response_pipeline_qwen(asr, model, ex, x_label)
                else:
                    pred = get_response_pipeline(asr, model, ex, x_label)
                print(pred)
                print(ex[y_label])
                scores = [
                    value[pred]
                    for value in cfm.get_scores(
                        ex[y_label], pred, ex["question"]
                    ).values()
                ]
                score = max(scores)
                print(score)
            except Exception as e:
                print(e)
                print(id)
                pred = "ERROR IN PROCESSING"
                score = "NA"
            if score != "NA":
                global_scores.append(score)
            pred_rep = '"""' + pred.replace("\n", "[NEW_LINE]") + '"""'
            f.write(f"{id}[SEP_DIAL]{pred_rep}[SEP_DIAL]{score}\n")
            if idx % 10 == 0 and idx != 0:
                f.flush()
    dial_scores[dial] = np.mean(global_scores)
    print(dial_scores)
print(dial_scores)
