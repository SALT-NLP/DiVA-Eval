cd eval
for i in pipe,meta-llama/Meta-Llama-3-8B-Instruct e2e,via e2e,salmonn_7b e2e,Qwen/Qwen-Audio-Chat pipe,Qwen/Qwen1.5-7B-Chat pipe,mistralai/Mistral-7B-Instruct-v0.2 pipe,meta-llama/Llama-2-7b-chat-hf;
do
    IFS=","
    set -- $i
    echo $1 $2
    python sdqa_eval.py $1 $2
done
