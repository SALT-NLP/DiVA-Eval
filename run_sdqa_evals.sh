cd eval
#git clone https://huggingface.co/WillHeld/WillHeld/llama3-via-v0
for i in e2e,via pipe,mistralai/Mistral-7B-Instruct-v0.2 e2e,salmonn_7b pipe,meta-llama/Meta-Llama-3-8B-Instruct e2e,Qwen/Qwen-Audio-Chat pipe,Qwen/Qwen1.5-7B-Chat pipe,meta-llama/Llama-2-7b-chat-hf;
do
    IFS=","
    set -- $i
    echo $1 $2
    python qa_eval.py $1 $2
done
