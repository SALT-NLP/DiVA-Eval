import copy
import os
import random
import sys

import xxhash
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
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    WhisperForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    AutoModel
)
from transformers.generation import GenerationConfig

anonymous = True


qwen_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-Audio-Chat", trust_remote_code=True
)
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

qwen2_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="auto"
)

qwen2_model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    trust_remote_code=True,
    do_sample=False,
    top_k=50,
    top_p=1.0,
)

diva_model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True)

# salmonn_model = SALMONN(
#     ckpt="./SALMONN_PATHS/salmonn_v1.pth",
#     whisper_path="./SALMONN_PATHS/whisper-large-v2",
#     beats_path="./SALMONN_PATHS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
#     vicuna_path="./SALMONN_PATHS/vicuna-13b-v1.1",
#     low_resource=False,
#     device="cuda:1",
# )
# salmonn_tokenizer = salmonn_model.llama_tokenizer

resampler = Audio(sampling_rate=16_000)

# @torch.no_grad
# def salmonn_fwd(audio_input, do_sample=False, temperature=0.001):
#     if audio_input == None:
#         return ""
#     sr, y = audio_input
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))
#     a = resampler.decode_example(
#         resampler.encode_example({"array": y, "sampling_rate": sr})
#     )
#     sf.write("tmp.wav", a["array"], a["sampling_rate"], format="wav")
#     streamer = TextIteratorStreamer(salmonn_tokenizer)
#     with torch.cuda.amp.autocast(dtype=torch.float16):
#         llm_message = salmonn_model.generate(
#             wav_path="tmp.wav",
#             prompt="You are a helpful assistant.",
#             do_sample=False,
#             top_p=1.0,
#             temperature=0.0,
#             device="cuda:1",
#             streamer=streamer,
#         )

#     response = ""
#     for new_tokens in streamer:
#         response += new_tokens
#         yield response.replace("</s>", "")


@torch.no_grad
def diva_audio(audio_input, do_sample=False, temperature=0.001):
    sr, y = audio_input
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    yield from diva_model.generate_stream(a["array"], None, do_sample=do_sample, max_new_tokens = 256)

@torch.no_grad
def qwen2_audio(audio_input, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": f"{x}.wav",
                },
            ],
        },
    ]
    text = qwen2_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = [
        librosa.load(f"{x}.wav", sr=qwen2_processor.feature_extractor.sampling_rate)[0]
    ]
    inputs = qwen2_processor(text=text, audios=audios, return_tensors="pt", padding=True)
    generate_ids = qwen2_model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = qwen2_processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


@torch.no_grad
def qwen_audio(audio_input, do_sample=False, temperature=0.001):
    if audio_input == None:
        return ""
    sr, y = audio_input
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
    query = qwen_tokenizer.from_list_format([{"audio": f"{x}.wav"}])

    response, history = qwen_model.chat(
        qwen_tokenizer,
        query=query,
        system="You are a helpful assistant.",
        history=None,
    )
    return response


def transcribe(audio_input, state, model_order):
    if audio_input == None:
        return (
            "",
            "",
            gr.Button(visible=False),
            gr.Button(visible=False),
            state,
        )

    def gen_from_via():
        via_resp = via(audio_input)
        for resp in via_resp:
            v_resp = gr.Textbox(
                value=resp,
                visible=True,
                label=model_names[0] if not anonymous else f"Model {order}",
            )
            yield (s_resp, q_resp)

    def gen_from_salmonn():
        salmonn_resp = salmonn_fwd(audio_input)
        for resp in salmonn_resp:
            s_resp = gr.Textbox(
                value=resp,
                visible=True,
                label=model_names[1] if not anonymous else f"Model {order}",
            )
            yield (s_resp, q_resp)

    def gen_from_qwen2():
        qwen_resp = qwen2_audio(audio_input)
        s_resp = gr.Textbox(
            value=qwen_resp,
            visible=True,
            label=model_names[2] if not anonymous else f"Model {order}",
        )
        yield (s_resp, q_resp)

    def gen_from_diva():
        diva_resp = diva_audio(audio_input)
        for resp in  diva_resp:
            d_resp = gr.Textbox(
                value=resp,
                visible=True,
                label=model_names[0] if not anonymous else f"Model {order}",
            )
            yield (s_resp, d_resp)

    def gen_from_qwen():
        qwen_resp = qwen_audio(audio_input)
        q_resp = gr.Textbox(
            value=qwen_resp,
            visible=True,
            label=model_names[2] if not anonymous else f"Model {order}",
        )
        yield (s_resp, q_resp)

    spinner_id = 0
    spinners = ["◐ ", "◓ ", "◑", "◒"]
    initial_responses = [("", "")]
    resp_generators = [
        gen_from_qwen2(),
        gen_from_diva(),
    ]
    order = -1
    resp_generators = [resp_generators[model_order[0]], resp_generators[model_order[1]]]
    for generator in [initial_responses, *resp_generators]:
        order += 1
        for resps in generator:
            s_resp, q_resp = resps
            resp_1 = resps[model_order[0]]
            resp_2 = resps[model_order[1]]
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
        gr.Button(visible=True),
        gr.Button(visible=True),
        responses_complete(state),
    )


def on_page_load(state, model_order):
    if state == 0:
        gr.Info(
            "Record something you'd say to an AI Assistant! Think about what you usually use Siri, Google Assistant, or ChatGPT for."
        )
        state = 1
        if anonymous:
            random.shuffle(model_order)
    return state, model_order


def recording_complete(state):
    if state == 1:
        gr.Info(
            "Once you submit your recording, you'll receive responses from different models. This might take a second."
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
    def clear(audio_input, model_order, pref_counter):
        if button_id != None:
            sr, y = audio_input
            db.insert(
                {
                    "audio_hash": xxhash.xxh32(bytes(y)).hexdigest(),
                    "best": model_shorthand[model_order[button_id]],
                }
            )
            pref_counter += 1
        counter_text = f"# {pref_counter}/10 Preferences Submitted"
        if pref_counter >= 10:
            code = "PLACEHOLDER"
            counter_text = f"# Completed! Completion Code: {code}"
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
            None,
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            pref_counter,
            counter_text
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

model_names = ["Qwen2 Audio", "DiVA"]
model_shorthand = ["qwen2", "diva"]
with gr.Blocks(theme=theme) as demo:
    submitted_preferences = gr.State(0)
    state = gr.State(0)
    model_order = gr.State([0, 1])
    with gr.Row():
        counter_text = gr.Markdown("# 0/10 Preferences Submitted.\n Follow the pop-up tips to submit your first preference.")
    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"], streaming=False, label="Audio Input"
        )

    with gr.Row():
        btn = gr.Button(value="Record Audio to Submit!", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            out1 = gr.Textbox(visible=False)
        with gr.Column(scale=1):
            out2 = gr.Textbox(visible=False)

    with gr.Row():
        best1 = gr.Button(value="This response is best", visible=False)
        best2 = gr.Button(value="This response is best", visible=False)

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
        inputs=[audio_input, state, model_order],
        outputs=[btn, out1, out2, best1, best2, state],
    )
    best1.click(
        fn=clear_factory(0),
        inputs=[audio_input, model_order, submitted_preferences],
        outputs=[model_order, btn, best1, best2, audio_input, out1, out2, submitted_preferences, counter_text],
    )
    best2.click(
        fn=clear_factory(1),
        inputs=[audio_input, model_order, submitted_preferences],
        outputs=[model_order, btn, best1, best2, audio_input, out1, out2, submitted_preferences, counter_text],
    )
    audio_input.clear(
        clear_factory(None),
        [audio_input, model_order, submitted_preferences],
        [model_order, btn, best1, best2, audio_input, out1, out2, submitted_preferences, counter_text],
    )
    demo.load(
        fn=on_page_load, inputs=[state, model_order], outputs=[state, model_order]
    )

demo.launch(share=True)
