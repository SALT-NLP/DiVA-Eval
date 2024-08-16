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
        virt_whisper_tokens = self.decoder(inputs_embeds=query_tokens, encoder_hidden_states=x)
        if self.projection.weight.shape[-1] == 5120:
            virtual_tokens = self.projection(virt_whisper_tokens[0].reshape(112, 5120))
        else:
            virtual_tokens = self.projection(virt_whisper_tokens[0])
        return virtual_tokens.to("cuda")


class VIA(nn.Module):
    def __init__(self, via_path):
        super().__init__()
        whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda").eval()
        connector = WhisperConnector()
        with open(via_path, "rb") as f:
            sd = load(f.read())

        with torch.no_grad():
            connector.query_tokens = nn.Parameter(sd["query_tokens"])
            connector.projection.weight = nn.Parameter(sd["projection.weight"].T)
            connector.projection.bias = nn.Parameter(sd["projection.bias"])
            connector.decoder = copy.deepcopy(whisper.model.decoder)
            wsd = {key.replace("connector.", ""): sd[key] for key in sd if key.startswith("connector.")}
            connector.decoder.load_state_dict(wsd)
        self.connector = connector.to("cuda")
        self.whisper_encoder = whisper.model.encoder.to("cuda")
        num_layers = 32
        num_gpus = 1
        device_map = dict(
            **{"model.embed_tokens": 1, "model.norm": 1, "lm_head": 2},
            **{"model.layers." + str(i): 1 + (i // (num_layers // num_gpus)) for i in range(num_layers)},
        )
        self.llama_decoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        self.tokenizer = AutoTokenizer.from_pretrained("WillHeld/via-llama")
        self.prefix = torch.tensor([128000, 128006, 882, 128007, 271]).to("cuda")
        self.pre_user_suffix = torch.tensor(
            self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n")
        ).to("cuda")
        self.final_header = torch.tensor([128009, 128006, 78191, 128007, 271]).to("cuda")

    def pad_or_trim(self, input_features, target_length=3000):
        current_length = input_features.shape[-1]
        if current_length < target_length:
            padding = target_length - current_length
            padded_features = torch.zeros(
                *input_features.shape[:-1], target_length, device=input_features.device, dtype=input_features.dtype
            )
            padded_features[..., :current_length] = input_features
            return padded_features
        elif current_length > target_length:
            return input_features[..., :target_length]
        return input_features

    def prepare_batch_inputs(self, audio, prompts, batch_size, padding=True):
        print(f"Audio shape before processing: {audio.shape if isinstance(audio, torch.Tensor) else 'not a tensor'}")
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=16_000, padding=True)
        input_features = inputs.input_features.to("cuda:0")
        print(f"Input features shape before padding: {input_features.shape}")

        input_features = self.pad_or_trim(input_features, target_length=3000)
        print(f"Input features shape after padding: {input_features.shape}")

        hidden_states = self.whisper_encoder(input_features=input_features)["last_hidden_state"]
        print(f"Hidden states shape: {hidden_states.shape}")

        virt_tokens = self.connector(hidden_states).squeeze()
        print(f"Virtual tokens shape: {virt_tokens.shape}")

        batch_size = virt_tokens.shape[0]

        if prompts != None and prompts != "":
            prefix_embeds = []
            for prompt in prompts:
                if prompt:
                    user_prompt_text = torch.tensor(
                        self.tokenizer(prompt, add_special_tokens=False)["input_ids"],
                        device=self.pre_user_suffix.device,
                    )
                    prefix = torch.cat([self.pre_user_suffix, user_prompt_text, self.prefix], axis=0)
                else:
                    prefix = self.prefix
                prefix_embeds.append(self.llama_decoder.model.embed_tokens(prefix))
            prefix_embeds = torch.stack(prefix_embeds)
        else:
            prefix_embeds = self.llama_decoder.model.embed_tokens(self.prefix).unsqueeze(0).repeat(batch_size, 1, 1)
        suffix_embeds = self.llama_decoder.model.embed_tokens(self.final_header).unsqueeze(0).repeat(batch_size, 1, 1)
        input_embeds = torch.cat([prefix_embeds, virt_tokens, suffix_embeds], dim=1)

        return input_embeds

    def generate(
        self,
        audio_batch: np.ndarray,
        prompts: list[str],
        return_log_probs: bool = False,
        temperature: float = 0.001,
        do_sample: bool = False,
        max_new_tokens: int = 128,
        padding: bool = True,
    ):
        input_embeds = self.prepare_batch_inputs(audio_batch, prompts, padding)

        batch_size = input_embeds.shape[0]
        outs = [[] for _ in range(batch_size)]
        log_probs = [[] for _ in range(batch_size)]
        outputs = None
        not_finished = torch.ones(batch_size, dtype=torch.bool, device="cuda:0")

        while not_finished.any() and max(len(out) for out in outs) < max_new_tokens:
            past_key_values = outputs.past_key_values if outputs else None
            outputs = self.llama_decoder(
                inputs_embeds=input_embeds.to("cuda:0").half(),
                return_dict=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            probs = F.softmax(next_token_logits / temperature, dim=-1)

            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1)

            if return_log_probs:
                token_log_probs = torch.log(probs)

            for i in range(batch_size):
                if not_finished[i]:
                    token = next_tokens[i].item()
                    outs[i].append(token)
                    if return_log_probs:
                        log_probs[i].append(token_log_probs[i, token].item())
                    if next_tokens[i] == 128009:  # EOT token
                        not_finished[i] = False

            next_embeds = self.llama_decoder.model.embed_tokens(next_tokens.unsqueeze(1))
            input_embeds = next_embeds

        decoded_outputs = []
        for out in outs:
            decoded = self.tokenizer.decode(out, skip_special_tokens=True).replace("<|eot_id|>", "")
            decoded_outputs.append(decoded)

        return outs, decoded_outputs, log_probs
