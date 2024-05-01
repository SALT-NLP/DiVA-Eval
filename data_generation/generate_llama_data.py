from time import sleep

from datasets import load_dataset
from huggingface_hub import InferenceClient
from ratelimit import limits, sleep_and_retry
from transformers import AutoTokenizer

dataset = load_dataset("yijingwu/HeySQuAD_human", split="train")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct")


CALLS = 240
RATE_LIMIT = 60


@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def create_distill_data(ex):
    chat = [
        {"role": "user", "content": ex["question"]},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    try:
        response = client.text_generation(prompt, max_new_tokens=24)
    except:
        sleep(100)
        return create_distill_data(ex)
    ex["response"] = response
    return ex


length = len(dataset)
splits = 14
step_size = length // splits
for start in range(splits):
    ds_processed = dataset.select(
        range((start * step_size), (start * step_size) + step_size)
    )

    ds_processed = ds_processed.map(create_distill_data)
    ds_processed.push_to_hub("WillHeld/HeySQuAD_distill", split="train." + str(start))
