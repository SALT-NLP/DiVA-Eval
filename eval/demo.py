import copy
import os
import random
import sys

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from accelerate import infer_auto_device_map
from datasets import Audio
from models.salmonn import SALMONN
from safetensors.torch import load, load_model
from tinydb import TinyDB
from torch import nn
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          LlamaForCausalLM, TextIteratorStreamer,
                          WhisperForConditionalGeneration)
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("WillHeld/via-llama")
prefix = torch.tensor([128000, 128006, 882, 128007, 271]).to("cuda:0")
pre_user_suffix = torch.tensor([271]).to("cuda:0")
final_header = torch.tensor([128009, 128006, 78191, 128007, 271]).to("cuda:0")
cache = None
anonymous = True


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
        return virtual_tokens.to("cuda:0")


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
    "./llama3-via-v0/model-00001-of-00004.safetensors",
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
# device_map = dict(
#    **{"model.embed_tokens": 1, "model.norm": 1, "lm_head": 2},
#    **{
#        "model.layers." + str(i): 1 + (i // (num_layers // num_gpus))
#        for i in range(num_layers)
#    },
# )
llama_decoder = LlamaForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16,
)
# Dynamic Shape makes the decoder compile slow?
# llama_decoder = torch.compile(llama_decoder)
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-Audio-Chat", trust_remote_code=True
)
device_map = {
    **{
        "transformer.wte": 1,
        "transformer.audio": 1,
        "transformer.ln_f": 1,
        "lm_head": 1,
    },
    **{"transformer.h." + str(i): 1 for i in range(num_layers)},
}
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-Audio-Chat",
    device_map="auto",
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


salmonn_model = SALMONN(
    ckpt="./SALMONN_PATHS/salmonn_v1.pth",
    whisper_path="./SALMONN_PATHS/whisper-large-v2",
    beats_path="./SALMONN_PATHS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    vicuna_path="./SALMONN_PATHS/vicuna-13b-v1.1",
    low_resource=False,
    device="cuda:1",
)
salmonn_tokenizer = salmonn_model.llama_tokenizer


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
        yield tokenizer.decode(outs, skip_special_tokens=True).replace("<|eot_id|>", "")
    return tokenizer.decode(outs, skip_special_tokens=True).replace("<|eot_id|>", "")


@torch.no_grad
def salmonn_fwd(audio_input, prompt, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    sf.write("tmp.wav", a["array"], a["sampling_rate"], format="wav")
    streamer = TextIteratorStreamer(salmonn_tokenizer)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = salmonn_model.generate(
            wav_path="tmp.wav",
            prompt=prompt,
            do_sample=False,
            top_p=1.0,
            temperature=0.0,
            device="cuda:1",
            streamer=streamer,
        )

    response = ""
    for new_tokens in streamer:
        response += new_tokens
        yield response.replace("</s>", "")


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
    query = qwen_tokenizer.from_list_format([{"audio": "tmp.wav"}, {"text": prompt}])

    response, history = qwen_model.chat(
        qwen_tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
    )
    return response


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
            inputs_embeds=inputs_embeds.to("cuda:0").half(),
            return_dict=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
        )
        global cache
        if cache is not None and i == 0:
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
        yield tokenizer.decode(outs).replace("<|eot_id|>", "")
    return tokenizer.decode(outs).replace("<|eot_id|>", "")


def transcribe(audio_input, text_prompt, state, model_order):
    print("test")
    if audio_input == None:
        return (
            "",
            "",
            "",
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            state,
        )

    def gen_from_via():
        via_resp = via(audio_input, text_prompt)
        for resp in via_resp:
            v_resp = gr.Textbox(value=resp, visible=True, label=model_names[0] if not anonymous else f"Model {order}")
            yield (v_resp, s_resp, q_resp)

    def gen_from_salmonn():
        salmonn_resp = salmonn_fwd(audio_input, text_prompt)
        for resp in salmonn_resp:
            s_resp = gr.Textbox(value=resp, visible=True, label=model_names[1] if not anonymous else f"Model {order}")
            yield (v_resp, s_resp, q_resp)

    def gen_from_qwen():
        qwen_resp = qwen_audio(audio_input, text_prompt)
        q_resp = gr.Textbox(value=qwen_resp, visible=True, label=model_names[2] if not anonymous else f"Model {order}")
        yield (v_resp, s_resp, q_resp)

    spinner_id = 0
    spinners = ["◐ ", "◓ ", "◑", "◒"]
    initial_responses = [("", "", "")]
    resp_generators = [
        gen_from_via(),
        gen_from_salmonn(),
        gen_from_qwen(),
    ]
    order = -1
    resp_generators = [resp_generators[model_order[0]], resp_generators[model_order[1]], resp_generators[model_order[2]]]
    for generator in [initial_responses, *resp_generators]:
        order += 1
        for resps in generator:
            v_resp, s_resp, q_resp = resps
            resp_1 = resps[model_order[0]]
            resp_2 = resps[model_order[1]]
            resp_3 = resps[model_order[2]]
            spinner = spinners[spinner_id]
            spinner_id = (spinner_id + 1) % 4
            yield (
                gr.Button(
                    value=spinner + " Generating Responses " + spinner,
                    interactive=False,
                    variant="primary",
                ),
                resp_1,
                resp_2,
                resp_3,
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                state,
            )
    yield (
        gr.Button(
            value="Click to compare models!", interactive=True, variant="primary"
        ),
        resp_1,
        resp_2,
        resp_3,
        gr.Button(visible=True),
        gr.Button(visible=True),
        gr.Button(visible=True),
        responses_complete(state),
    )


def on_page_load(state, model_order):
    if state == 0:
        gr.Info(
            "Record what you want to say to your AI Assistant! All Audio recordings are stored only temporarily and will be erased as soon as you exit this page."
        )
        state = 1
        if anonymous:
            random.shuffle(model_order)
    return state, model_order


def recording_complete(state):
    if state == 1:
        gr.Info(
            "Submit your recording to get responses from all three models! You can also influence the model responses with an optional prompt."
        )
        state = 2
    return (
        gr.Button(
            value="Click to compare models!", interactive=True, variant="primary"
        ),
        state,
    )


def responses_complete(state):
    if state == 2:
        gr.Info(
            "Give us your feedback! Mark which model gave you the best response so we can understand the quality of these different voice assistant models."
        )
        state = 3
    return state


def clear_factory(button_id):
    def clear(audio_input, text_prompt, model_order):
        if button_id != None:
            sr, y = audio_input
            db.insert(
                {
                    "audio_hash": hash(str(y)),
                    "text_prompt": text_prompt,
                    "best": model_shorthand[model_order[button_id]],
                }
            )
        if anonymous:
            random.shuffle(model_order)
        return (
            model_order,
            gr.Button(
                value="Record Audio to Submit!",
                interactive=False,
            ),
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            None,
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
        )

    return clear


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

db = TinyDB("user_study.json")

model_names = ["Llama 3 VIA", "SALMONN", "Qwen Audio"]
model_shorthand = ["via", "salmonn", "qwen"]
with gr.Blocks(theme=theme) as demo:
    state = gr.State(0)
    model_order = gr.State([0, 1, 2])
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], streaming=False, label="Audio Input"
        )
    with gr.Row():
        prompt = gr.Textbox(
            value="",
            label="Text Prompt",
            placeholder="Optional: Additional text prompt to influence how the model responds to your speech.",
        )

    with gr.Row():
        btn = gr.Button(value="Record Audio to Submit!", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            out1 = gr.Textbox(visible=False)
        with gr.Column(scale=1):
            out2 = gr.Textbox(visible=False)
        with gr.Column(scale=1):
            out3 = gr.Textbox(visible=False)

    with gr.Row():
        best1 = gr.Button(value="This response is best", visible=False)
        best2 = gr.Button(value="This response is best", visible=False)
        best3 = gr.Button(value="This response is best", visible=False)

    audio_input.stop_recording(
        recording_complete,
        [state],
        [btn, state],
    )
    audio_input.start_recording(
        lambda: gr.Button(
            value="Uploading Audio to Cloud", interactive=False, variant="primary"
        ),
        None,
        btn,
    )
    btn.click(
        fn=transcribe,
        inputs=[audio_input, prompt, state, model_order],
        outputs=[btn, out1, out2, out3, best1, best2, best3, state],
    )
    best1.click(
        fn=clear_factory(0),
        inputs=[audio_input, prompt, model_order],
        outputs=[model_order,btn, best1, best2, best3, audio_input, out1, out2, out3],
    )
    best2.click(
        fn=clear_factory(1),
        inputs=[audio_input, prompt, model_order],
        outputs=[model_order,btn, best1, best2, best3, audio_input, out1, out2, out3],
    )
    best3.click(
        fn=clear_factory(2),
        inputs=[audio_input, prompt, model_order],
        outputs=[model_order,btn, best1, best2, best3, audio_input, out1, out2, out3],
    )
    audio_input.clear(
        clear_factory(None),
        [audio_input, prompt, model_order],
        [model_order, btn, best1, best2, best3, audio_input, out1, out2, out3],
    )
    demo.load(fn=on_page_load, inputs=[state, model_order], outputs=[state, model_order])

demo.launch(share=False)
