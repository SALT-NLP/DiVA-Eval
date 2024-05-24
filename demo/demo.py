import copy

import soundfile as sf
import gradio as gr
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio
from safetensors.torch import load, load_model
from torch import nn
from accelerate import infer_auto_device_map
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    WhisperForConditionalGeneration,
)
from transformers.generation import GenerationConfig
import os

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
with open(
    "/data/wheld3/audio/levanter/via-7b-3/model-00001-of-00004.safetensors",
    "rb",  # "/data/wheld3/audio/levanter/step-4299/model-00001-of-00004.safetensors", "rb"
) as f:
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
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-Audio-Chat", trust_remote_code=True
)
device_map = {
    **{
        "transformer.wte": 3,
        "transformer.audio": 3,
        "transformer.ln_f": 3,
        "lm_head": 3,
    },
    **{
        "transformer.h." + str(i): 3 + (i // (num_layers // num_gpus))
        for i in range(num_layers)
    },
}
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-Audio-Chat",
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).eval()

qwen_model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-Audio-Chat",
    trust_remote_code=True,
    do_sample=False,
    top_k=50,
    top_p=1.0,
)


@torch.no_grad
def pipeline(audio_input, prompt, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
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
        suffix = torch.cat([pre_user_suffix, user_prompt_text, final_header], axis=0)
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
            print(
                torch.cdist(
                    cache.double(),
                    outputs.hidden_states[-1][-1, -1]
                    .unsqueeze(0)
                    .to("cuda:0")
                    .double(),
                )
            )
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
        yield tokenizer.decode(outs)
    return tokenizer.decode(outs)


@torch.no_grad
def qwen_audio(audio_input, prompt, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    sf.write("tmp.wav", a["array"], a["sampling_rate"], format="wav")
    query = qwen_tokenizer.from_list_format(
        [{"audio": "tmp.wav"}, {"text": "Give a simple one sentence answer."}]
    )

    response, history = qwen_model.chat(
        qwen_tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
    )
    yield from response


@torch.no_grad
def via(audio_input, prompt, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
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
        suffix = torch.cat([pre_user_suffix, user_prompt_text, final_header], axis=0)
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
            print(
                torch.cdist(
                    cache.double(),
                    outputs.hidden_states[-1][-1, -1]
                    .unsqueeze(0)
                    .to("cuda:0")
                    .double(),
                )
            )
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
        yield tokenizer.decode(outs)
    return tokenizer.decode(outs)

<<<<<<< Updated upstream
    for resp in tok_gen:
        yield resp


demo = gr.Interface(
    transcribe,
    [
        "state",
        gr.Audio(sources=["upload", "microphone"], streaming=False),
        gr.Textbox(value=""),
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

def transcribe(audio_input, text_prompt):
    print("test")
    pipeline_resp = pipeline(audio_input, text_prompt)
    via_resp = via(audio_input, text_prompt)
    qwen_resp = qwen_audio(audio_input, text_prompt)

    for resp in via_resp:
        v_resp = resp
        yield v_resp, "", ""
    for resp in pipeline_resp:
        p_resp = resp
        yield v_resp, p_resp, ""
    for resp in qwen_resp:
        q_resp = resp
        yield v_resp, p_resp, q_resp


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        "Record what you have to say, prompt the model with some text instructions, and click run to see the responses!"
    )
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], streaming=False, label="Audio Input"
        )
        prompt = gr.Textbox(value="", label="Text Prompt")
    with gr.Row():
        out1 = gr.Textbox(label="Llama 3 VIA")
        out2 = gr.Textbox(label="Whisper + Llama 3")
    with gr.Row():
        out3 = gr.Textbox(label="Qwen Audio")
    btn = gr.Button("Run")
    btn.click(fn=transcribe, inputs=[audio_input, prompt], outputs=[out1, out2, out3])
# demo = gr.Interface(
#     transcribe,
#     [
#         "state",
#         gr.Audio(sources=["microphone"], streaming=False),
#         gr.Textbox(value=""),
#     ],
#     ["state", "text", "text"],
#     theme=theme,
#     flagging_options=["Option 1 Preferred", "Option 2 Preferred"],
# )

demo.launch(share=True)
