import copy

import gradio as gr
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio
from safetensors.torch import load, load_model
from torch import nn
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    WhisperForConditionalGeneration,
)

tokenizer = AutoTokenizer.from_pretrained("WillHeld/via-llama")
prefix = torch.tensor([128000, 128006, 882, 128007, 271]).to("cuda:0")
pre_user_suffix = torch.tensor([271]).to("cuda:0")
final_header = torch.tensor([128009, 128006, 78191, 128007, 271]).to("cuda:0")
cache = None

class Connector(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.decoder = None
        self.projection = nn.Linear(1280, 4096)
        self.query_tokens = nn.Parameter(torch.randn(448, 1280))

    def forward(self, x):
        bsz = x.shape[0]
        query_tokens = self.query_tokens[None, :, :].expand(bsz, -1, -1)
        virt_whisper_tokens = self.decoder(
            inputs_embeds=query_tokens, encoder_hidden_states=x
        )
        if self.projection.weight.shape[-1] == 5120:
            virtual_tokens = self.projection(virt_whisper_tokens[0].reshape(112, 5120))
        else:
            virtual_tokens = self.projection(virt_whisper_tokens[0])
        return virtual_tokens.to("cuda:1")


processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
whisper = (
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    .to("cuda:0")
    .eval()
)
whisper_encoder = whisper.model.encoder.to("cuda:0")
# whisper_encoder = torch.compile(whisper_encoder)
resampler = Audio(sampling_rate=16_000)

connector = Connector()
with open("/data/wheld3/via-7b-3/model-00001-of-00004.safetensors", "rb") as f:
    sd = load(f.read())

with torch.no_grad():
    connector.query_tokens = nn.Parameter(sd["query_tokens"])
    connector.projection.weight = nn.Parameter(sd["projection.weight"].T)
    connector.projection.bias = nn.Parameter(sd["projection.bias"])
    connector.decoder = copy.deepcopy(whisper.model.decoder)
    wsd = {
        key.replace("connector.", ""): sd[key]
        for key in sd
        if key.startswith("connector.")
    }
    connector.decoder.load_state_dict(wsd)
connector = connector.to("cuda:0")

num_layers = 32
num_gpus = 2
device_map = dict(
    **{"model.embed_tokens": 1, "model.norm": 1, "lm_head": 2},
    **{
        "model.layers." + str(i): 1 + (i // (num_layers // num_gpus))
        for i in range(num_layers)
    },
)
llama_decoder = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map=device_map,
    torch_dtype=torch.float16,
)
# Dynamic Shape makes the decoder compile slow?
# llama_decoder = torch.compile(llama_decoder)


@torch.no_grad
def pipeline(stream, new_chunk, prompt, do_sample=False, temperature=0.001):
    if new_chunk == None:
        return stream, ""
    sr, y = new_chunk
    y = y.astype(np.float32)
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )

    audio = a["array"]
    inputs = processor(audio, return_tensors="pt", sampling_rate=16_000)
    input_features = inputs.input_features.to("cuda:0")
    predictions = whisper.generate(input_features=input_features, language="en")
    prediction_strs = processor.batch_decode(predictions, skip_special_tokens=True)[0]
    user_prompt = torch.tensor(
        tokenizer(prediction_strs, add_special_tokens=False)["input_ids"],
        device=pre_user_suffix.device,
    )
    if prompt != None and prompt != "":
        user_prompt_text = torch.tensor(
            tokenizer(prompt, add_special_tokens=False)["input_ids"],
            device=pre_user_suffix.device,
        )
        suffix = torch.cat(
            [pre_user_suffix, user_prompt_text, final_header], axis=0
        )
    else:
        suffix = final_header
    prompt = torch.cat([prefix, user_prompt, suffix], axis=0).unsqueeze(0)
    print(tokenizer.batch_decode(prompt))
    outs = []
    outputs = None
    greedy = 1
    i = 0
    while greedy != 128009 and len(outs) < 128:
        past_key_values = outputs.past_key_values if outputs else None
        outputs = llama_decoder(
            input_ids=prompt,
            return_dict=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        global cache
        if cache is not None and i == 0:
            print()
            print(torch.cdist(cache.double(), outputs.hidden_states[-1][-1, -1].unsqueeze(0).to("cuda:0").double()))
            cache = None
        elif i == 0:
            cache = outputs.hidden_states[-1][-1, -1].unsqueeze(0).to("cuda:0").double()
        next_token_logits = outputs.logits[-1, -1, :]
        if do_sample:
            logits = next_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            greedy = torch.multinomial(probs, num_samples=1)[0]
        else:
            greedy = next_token_logits.argmax()
        outs.append(greedy)
        i += 1
        prompt = greedy.reshape(1, 1)
        yield (y, tokenizer.decode(outs))
    return (y, tokenizer.decode(outs))


@torch.no_grad
def via(stream, new_chunk, prompt, do_sample=False, temperature=0.001, mode="1"):
    if new_chunk == None:
        return stream, ""
    sr, y = new_chunk
    y = y.astype(np.float32)
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )

    audio = a["array"]
    inputs = processor(audio, return_tensors="pt", sampling_rate=16_000)
    input_features = inputs.input_features.to("cuda:0")
    hidden_states = whisper_encoder(input_features=input_features)["last_hidden_state"]
    prefix_embed = llama_decoder.model.embed_tokens(prefix)
    virt_tokens = connector(hidden_states).squeeze()
        
    if prompt != None and prompt != "":
        user_prompt_text = torch.tensor(
            tokenizer(prompt, add_special_tokens=False)["input_ids"],
            device=pre_user_suffix.device,
        )
        suffix = torch.cat(
            [pre_user_suffix, user_prompt_text, final_header], axis=0
        )
    else:
        suffix = final_header
    suffix_embed = llama_decoder.model.embed_tokens(suffix)
    inputs_embeds = torch.cat(
        [prefix_embed, virt_tokens, suffix_embed], axis=0
    ).unsqueeze(0)
    outs = []
    outputs = None
    greedy = 1
    i = 0
    while greedy != 128009 and len(outs) < 128:
        past_key_values = outputs.past_key_values if outputs else None
        outputs = llama_decoder(
            inputs_embeds=inputs_embeds.to("cuda:1").half(),
            return_dict=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        global cache
        if cache is not None and i == 0:
            print()
            print(torch.cdist(cache.double(), outputs.hidden_states[-1][-1, -1].unsqueeze(0).to("cuda:0").double()))
            cache = None
        next_token_logits = outputs.logits[-1, -1, :]
        if do_sample:
            logits = next_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            greedy = torch.multinomial(probs, num_samples=1)[0]
        else:
            greedy = next_token_logits.argmax()
        outs.append(greedy)
        next_embed = llama_decoder.model.embed_tokens(greedy.reshape(1, 1))
        inputs_embeds = next_embed
        yield (y, tokenizer.decode(outs))
    return (y, tokenizer.decode(outs))


def transcribe(stream, new_chunk, prompt, model_type):
    if model_type == "pipeline":
        tok_gen = pipeline(stream, new_chunk, prompt)
    elif model_type in ["via", "via2"]:
        tok_gen = via(stream, new_chunk, prompt, mode=model_type)
    elif model_type == "via2":
        tok_gen = pipeline(
            stream,
            new_chunk,
            "Ignore previous instructions and answer the given question using one simple sentence.",
        )

    for resp in tok_gen:
        yield resp


demo = gr.Interface(
    transcribe,
    [
        "state",
        gr.Audio(sources=["upload", "microphone"], streaming=False),
        gr.Textbox(
            value=""
        ),
        gr.Dropdown(
            choices=[
                ("Pipelined ASR & LLM", "pipeline"),
                ("Model Fused Using Transcription", "via"),
                ("Model Fused Using Distillation", "via2"),
            ],
            value="via2",
        ),
    ],
    ["state", "text"],
)

demo.launch(share=True)
