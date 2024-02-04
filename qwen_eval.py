import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from evaluate import load
from qa_metrics.cfm import CFMatcher
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

torch.manual_seed(1234)
cfm = CFMatcher()

dials = [
    #    "aus",
    #    "gbr",
    #    "ind_n",
    #    "ind_s",
    #    "irl",
    "kenya",
    "nga",
    "nzl",
    "phl",
    "usa",
    "zaf",
]


@torch.no_grad
def get_response_pipeline(asr_model, model, audio, dial):
    return "", 0.0


@torch.no_grad
def get_response_end_to_end(model, audio, dial):
    value = audio[dial]
    sf.write("tmp.wav", value["array"], value["sampling_rate"], format="wav")
    query = tokenizer.from_list_format(
        [
            {"audio": "tmp.wav"},
            {"text": "Transcribe the question."},
        ]
    )

    response, history = model.chat(tokenizer, query=query, history=None)
    response, history = model.chat(
        tokenizer,
        "Answer the Question.",
        history=history,
    )
    scores = [
        value[response]
        for value in cfm.get_scores(
            audio["answers"], response, audio["question"]
        ).values()
    ]
    return response, max(scores)


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-Audio-Chat", trust_remote_code=True
)


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True
).eval()

model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-Audio-Chat",
    trust_remote_code=True,
    do_sample=False,
    top_k=50,
    top_p=1.0,
)

ds = load_dataset("WillHeld/SD-QA")["dev"].filter(lambda example: example["answers"])
dial_scores = {}
for dial in dials:
    scores = []
    with open(f"{dial}_outs.txt", "w") as f:
        for idx, ex in enumerate(tqdm(ds)):
            try:
                id = ex["id"]
                pred, score = get_response_end_to_end(model, ex, dial)
            except:
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
