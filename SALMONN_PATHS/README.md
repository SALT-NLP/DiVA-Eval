For SALMONN-13B v1, you need to use the following dependencies:
2. Download [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`.
4. Download [vicuna 13B v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1/tree/main) to ```vicuna_path```.
5. Download [salmonn v1](https://huggingface.co/tsinghua-ee/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt_path```.
6. Running with ```python3 cli_inference.py --ckpt_path xxx --whisper_path xxx --beats_path xxx --vicuna_path xxx``` in A100-SXM-80GB. Now you can input ```wav_path``` and ```prompt```. Enjoy yourself !
