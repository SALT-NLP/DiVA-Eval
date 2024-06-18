import argparse
import os

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qa_metrics.cfm import CFMatcher
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    PrefixConstrainedLogitsProcessor,
    pipeline,
)
from transformers.generation import GenerationConfig

from load_via_eval import load_via_eval
from models.salmonn import SALMONN
from models.via import VIA

torch.manual_seed(1234)

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


def label_forcing(labels):
    add_spaces = [
        len(tokenizer(" " + label).input_ids) == len(tokenizer(label).input_ids)
        for label in labels
    ]
    labels = [" " + label if False else label for i, label in enumerate(labels)]
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


dataset_name = args.dataset_name
dataset_config = {
    "Mustard_sarcasm": (
        {"Yes": True, "No": False},
        "Does the tone or content of the previous statement indicate sarcasm?",
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
        "What emotion does the previous statement communicate?\nIf there is no clear emotion, respond 'Neutral'.",
    ),
    "URFunny_humor": (
        {"Yes": "humor", "No": "not_humor"},
        "Is the final sentence of the previous passage a punchline to a joke?",
    ),
    "Callhome_relationships": (
        {"Family": "RELATIVE", "Friends": "FRIEND"},
        "Are the people speaking family or friends?",
    ),
    "IEMOCAP_emotion_recognition": (
        {
            "Anger": "angry",
            "Happiness": "happy",
            "Neutral": "neutral",
            "Sadness": "sad",
        },
        "What emotion does the previous statement communicate?\nIf there is no clear emotion, respond 'Neutral'.",
    ),
}

if "via" in args.model_name.lower() or "llama" in args.model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained("WillHeld/via-llama")
elif "salmonn" in args.model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
elif "qwen" in args.model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-Audio-Chat", trust_remote_code=True
    )


label_map = dataset_config[dataset_name][0]
prompt = dataset_config[dataset_name][1]
labels = list(label_map.keys())
forcing_fn, trunc_labels = label_forcing(labels)
for i, label in enumerate(labels):
    val = label_map[label]
    del label_map[label]
    label_map[trunc_labels[i]] = val
labels = list(label_map.keys())
print(label_map)
dials = ["default"]
dial_scores = {}
for dial in dials:
    x_label, y_label, ds = load_via_eval(dataset_name, language=dial)
    label_relevant = {}
    true = []
    preds = []
    name_short = model_name.lower().split("/")[-1]
    filename = f"./{dataset_name}_tmp_Results/{m_type}_{name_short}/{dial}_outs.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
        for idx, ex in enumerate(tqdm(ds)):
            line = lines[idx]
            f_idx, pred, score = line.split("[SEP_DIAL]")
            if "ERROR IN PROCESSING" in pred:
                pred = "neutral"
            else:
                pred = label_map[pred.replace('"""', "").strip()]
            assert idx == int(f_idx), (idx, f_idx)
            preds.append(pred)
            true.append(ex[y_label])
if len(labels) > 2:
    print(f1_score(true, preds, average="weighted"))
else:
    print(f1_score(true, preds, average="binary", pos_label=label_map[labels[0]]))
