cd eval
#git clone https://huggingface.co/WillHeld/llama3-via-v0
#e2e,via e2e,salmonn e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python qa_eval.py $1 $2 --dataset_name "Spoken_Dialect_QA"
# done

# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python qa_eval.py $1 $2 --dataset_name "non_social_HeySquad_QA"
# done

# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python classification_eval.py $1 $2
# done

for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
do
    IFS=","
    set -- $i
    echo $1 $2
    python classification_eval.py $1 $2 --dataset_name "URFunny_humor"
done

# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python classification_eval.py $1 $2 --dataset_name "MELD_emotion_recognition"
# done

# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python classification_eval.py $1 $2 --dataset_name "IEMOCAP_emotion_recognition"
# done

# for i in e2e,via e2e,salmonn; #e2e,Qwen/Qwen-Audio-Chat pipe,meta-llama/Meta-Llama-3-8B-Instruct;
# do
#     IFS=","
#     set -- $i
#     echo $1 $2
#     python classification_eval.py $1 $2 --dataset_name "Callhome_relationships"
# done
