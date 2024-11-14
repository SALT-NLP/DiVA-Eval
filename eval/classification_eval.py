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
    PrefixConstrainedLogitsProcessor,
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
            "content": "You are a helpful assistant. " + prompt,
        },
        {"role": "user", "content": text},
        {"role": "assistant", "content": ""},
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

    output = model.generate(
        query, max_new_tokens=1, logits_processor=[logits_processor]
    )
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


def label_forcing(labels):
    if hasattr(tokenizer, "tokenizer"):
        tokens = [tokenizer.tokenize(label) for label in labels]
        label_tokens = [
            tokenizer.convert_tokens_to_ids(token_label) for token_label in tokens
        ]
    else:
        label_tokens = tokenizer.batch_encode_plus(
            labels, add_special_tokens=False
        ).input_ids

    for label_token in label_tokens:
        if len(label_token) > 1:
            print("WARNING: Label Is Multiple Tokens, Only Using the First")

    label_tokens = [label_token[0] for label_token in label_tokens]

    assert len(set(label_tokens)) == len(label_tokens), "Labels are Not Unique"

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return label_tokens

    return prefix_allowed_tokens_fn, tokenizer.batch_decode(label_tokens)


@torch.no_grad
def get_response_end_to_end_s(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_s.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            wav_path="tmp_s.wav",
            prompt=prompt,
            num_beams=1,
            do_sample=False,
            top_p=1.0,
            logits_processor=logits_processor,
            max_new_tokens=1,
        )
    response = llm_message[0]

    return response

@torch.no_grad
def get_response_blsp(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_blsp_c.wav", value["array"], value["sampling_rate"], format="wav")
    with torch.cuda.amp.autocast(dtype=torch.float16):
        instruction = prompt
        input_ids = tokenizer(
            f"###[Human]:{instruction}", return_tensors="pt"
        ).input_ids.cuda()
        audio = "tmp_blsp_c.wav"
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
            logits_processor=logits_processor,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
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
                    "text": prompt,
                },
            ],
        },
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = [
        librosa.load("tmp_q.wav", sr=processor.feature_extractor.sampling_rate)[0]
    ]
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    generate_ids = model.generate(
        **inputs, max_new_tokens=1, logits_processor=[logits_processor]
    )
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
            prompt="\n" + prompt,
            logits_processor=logits_processor,
            max_new_tokens=1,
        )
    response = llm_message

    return response


@torch.no_grad
def get_response_end_to_end_q(model, audio, dial):
    value = audio[dial]
    sf.write("tmp_q.wav", value["array"], value["sampling_rate"], format="wav")
    query = tokenizer.from_list_format(
        [
            {"audio": "tmp_q.wav"},
            {"text": "\n" + prompt},
        ]
    )

    response, history = model.chat(
        tokenizer,
        query=query,
        system="",
        history=None,
        logits_processor=[logits_processor],
        max_new_tokens=1 + 1,
    )
    return response


parser = argparse.ArgumentParser("sdqa_args")
parser.add_argument("m_type", help="path for transcript file", type=str)
parser.add_argument("model_name", help="path for transcript file", type=str)
parser.add_argument(
    "--dataset_name",
    help="path for transcript file",
    type=str,
    default="Mustard_sarcasm",
)
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
    if "Qwen2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
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
            do_sample=False,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            num_return_sequences=1,
        )
        generation_config.update(
            **{
                "max_new_tokens": 1,
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
        tokenizer = model.llama_tokenizer
    elif "via" in model_name:
        # model = VIA("../llama3-via-v0/model-00001-of-00004.safetensors")
        model = VIA("/data/wheld3/step-4299/model-00001-of-00004.safetensors")
        tokenizer = model.tokenizer

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
dataset_config = {
    "Mustard_sarcasm": (
        {"Yes": True, "No": False},
        "Respond whether the input is sarcastic. Answer with a simple yes or no.",
    ),
    "MELD_emotion_recognition": (
        {
            "Anger": "anger",
            "Disgust": "disgust",
            "Fear": "fear",
            "Joy": "joy",
            "Neutral": "neutral",
            "Sadness": "sadness",
            "Surprise": "surprise",
        },
        "Respond in a single word what emotion the input exhibits.\nIf there is no clear emotion, respond 'Neutral'.",
    ),
    "URFunny_humor": (
        {"Yes": "humor", "No": "not_humor"},
        "Respond whether the input is intended to be humorous. Answer with a simple yes or no.",
    ),
    "Callhome_relationships": (
        {"Family": "RELATIVE", "Friends": "FRIEND"},
        "Respond whether the people talking are family or friends. Respond simply 'Family' or 'Friends'.",
    ),
    "IEMOCAP_emotion_recognition": (
        {
            "Anger": "angry",
            "Happiness": "happy",
            "Neutral": "neutral",
            "Sadness": "sad",
        },
        "Respond in a single word what emotion the audio represents.\nIf there is no clear emotion, respond 'Neutral'.",
    ),
}

label_map = dataset_config[dataset_name][0]
prompt = dataset_config[dataset_name][1]
labels = list(label_map.keys())
forcing_fn, trunc_labels = label_forcing(labels)
for i, label in enumerate(labels):
    val = label_map[label]
    del label_map[label]
    label_map[trunc_labels[i]] = val
labels = list(label_map.keys())
logits_processor = PrefixConstrainedLogitsProcessor(
    prefix_allowed_tokens_fn=forcing_fn, num_beams=1
)
dials = ["default"]
dial_scores = {}
for dial in dials:
    x_label, y_label, ds = load_via_eval(dataset_name, language=dial)
    scores = []
    name_short = model_name.lower().split("/")[-1]
    filename = f"./{dataset_name}_tmp_Results/{m_type}_{name_short}/{dial}_outs.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for idx, ex in enumerate(tqdm(ds)):
            try:
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
                score = 1 if label_map[pred] == ex[y_label] else 0
            except Exception as e:
                print("ERROR")
                print(e)
                print(idx)
                pred = "ERROR IN PROCESSING"
                score = 0
            if score != "NA":
                scores.append(score)
            pred_rep = '"""' + pred.replace("\n", "[NEW_LINE]") + '"""'
            f.write(f"{idx}[SEP_DIAL]{pred_rep}[SEP_DIAL]{score}\n")
            if idx % 10 == 0 and idx != 0:
                f.flush()
    dial_scores[dial] = np.mean(scores)
    print(dial_scores)
print(dial_scores)
