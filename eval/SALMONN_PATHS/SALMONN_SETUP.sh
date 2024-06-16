git clone https://huggingface.co/tsinghua-ee/SALMONN/
mv SALMONN/salmonn_v1.pth .
rm -r SALMONN
git clone https://huggingface.co/lmsys/vicuna-13b-v1.1
git clone https://huggingface.co/openai/whisper-large-v2
wget https://huggingface.co/spaces/fffiloni/SALMONN-7B-gradio/resolve/677c0125de736ab92751385e1e8664cd03c2ce0d/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true
mv BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
