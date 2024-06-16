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


class WhisperConnector(nn.Module):
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


class VIA(nn.Module):
    def __init__(self, via_path):
        super().__init__()
        whisper = (
            WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            .to("cuda:0")
            .eval()
        )
        connector = WhisperConnector()
        with open(via_path, "rb") as f:
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
        self.connector = connector.to("cuda:0")
        self.whisper_encoder = whisper.model.encoder.to("cuda:0")
        num_layers = 32
        num_gpus = 2
        device_map = dict(
            **{"model.embed_tokens": 1, "model.norm": 1, "lm_head": 2},
            **{
                "model.layers." + str(i): 1 + (i // (num_layers // num_gpus))
                for i in range(num_layers)
            },
        )
        self.llama_decoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.tokenizer = AutoTokenizer.from_pretrained("WillHeld/via-llama")
        self.prefix = torch.tensor([128000, 128006, 882, 128007, 271]).to("cuda:0")
        self.pre_user_suffix = torch.tensor(
            self.tokenizer.encode(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            )
        ).to("cuda:0")
        self.final_header = torch.tensor([128009, 128006, 78191, 128007, 271]).to(
            "cuda:0"
        )

    def generate(
        self, audio, prompt, do_sample=False, logits_processor=None, max_new_tokens=128
    ):
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000)
        input_features = inputs.input_features.to("cuda:0")
        hidden_states = self.whisper_encoder(input_features=input_features)[
            "last_hidden_state"
        ]
        virt_tokens = self.connector(hidden_states).squeeze()

        if prompt != None and prompt != "":
            user_prompt_text = torch.tensor(
                self.tokenizer(prompt, add_special_tokens=False)["input_ids"],
                device=self.pre_user_suffix.device,
            )
            prefix = torch.cat(
                [self.pre_user_suffix, user_prompt_text, self.prefix], axis=0
            )
        else:
            prefix = self.prefix
        prefix_embed = self.llama_decoder.model.embed_tokens(prefix)
        suffix = self.final_header
        suffix_embed = self.llama_decoder.model.embed_tokens(suffix)
        inputs_embeds = torch.cat(
            [prefix_embed, virt_tokens, suffix_embed], axis=0
        ).unsqueeze(0)
        outs = []
        outputs = None
        greedy = 1
        i = 0
        while greedy != 128009 and len(outs) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llama_decoder(
                inputs_embeds=inputs_embeds.to("cuda:1").half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[-1, -1, :]

            if logits_processor:
                local_outs = torch.tensor(outs) if outs != [] else suffix
                local_outs = local_outs.reshape(1, -1)
                next_token_logits = logits_processor(
                    local_outs,
                    next_token_logits.reshape(1, -1),
                )
                next_token_logits = next_token_logits.flatten()
            if do_sample:
                logits = next_token_logits / temperature
                probs = F.softmax(logits, dim=-1)
                greedy = torch.multinomial(probs, num_samples=1)[0]
            else:
                greedy = next_token_logits.argmax()
            outs.append(greedy)
            next_embed = self.llama_decoder.model.embed_tokens(greedy.reshape(1, 1))
            inputs_embeds = next_embed
        return self.tokenizer.decode(outs, skip_special_tokens=True).replace(
            "<|eot_id|>", ""
        )
