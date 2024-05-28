from moviepy.editor import VideoFileClip
from datasets import Dataset
import pandas as pd
from datasets import Audio

import os
import shutil

def extract_audio(mp4_file, output_audio_file):
    video = VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(
        output_audio_file,
    )


for mp4_dir in tqdm(glob("test/test/*.mp4")):
    extract_audio(
            mp4_dir,
            mp4_dir.replace("mp4", "wav"),
    )
# Specify the source and target directories
source_directory = 'video_clip/'
target_directory = 'test/'

def filter_dict_by_keys(original_dict, keys_to_keep):
    return {key: value for key, value in original_dict.items() if key in keys_to_keep}


import json

file_path = 'GT_laughter_reason.json'

with open(file_path, 'r') as file:
        data = json.load(file)
print(data)


test_list=[
            "1_80",
            "1_1001",
            "1_10810",
            "1_10857",
            "1_10859",
            "1_11042",
            "1_11046",
            "1_11201",
            "1_11236",
            "1_11529",
            "1_11928",
            "1_1666",
            "1_1678",
            "1_1722",
            "1_2075",
            "1_2420",
            "1_2423",
            "1_2830",
            "1_3069",
            "1_3177",
            "1_3204",
            "1_3649",
            "1_3837",
            "1_3840",
            "1_4352",
            "1_4603",
            "1_4792",
            "1_4949",
            "1_5109",
            "2_590",
            "2_559",
            "1_8505",
            "1_9009",
            "2_225",
            "2_264",
            "2_268",
            "1_7494",
            "1_7341",
            "1_6860",
            "1_6645",
            1402,
            3956,
            2751,
            9984,
            8066,
            1150,
            9789,
            3812,
            2532,
            3049,
            6225,
            1197,
            7966,
            956,
            5072,
            2420,
            4236,
            1695,
            8241,
            3435,
            5251,
            6337,
            6240,
            8558,
            3809,
            4240,
            1057,
            6915,
            5532,
            3369,
            2807,
            620,
            6744,
            7880,
            3740,
            2750,
            5378,
            6575,
            8328,
            878
        ]
test_list2=[]
for i in test_list:
    test_list2.append(str(i))
print(test_list2)

new_data = filter_dict_by_keys(data, test_list2)

print(new_data)
print("File transfer complete.")

file_path = 'annotation.json'

# Open the file in write mode and write the dictionary to the file
with open(file_path, 'w') as file:
    json.dump(new_data, file, indent=4)

import json

with open("annotation.json") as fh:
    meta_data = json.load(fh)


data_dict = {
    "audio": [
        f"/test/{key}.wav"
        for key in sorted(meta_data.keys())
    ],
    "reason": [meta_data[key] for key in sorted(meta_data.keys())]
}
audio_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
print(audio_dataset)

#audio_dataset.push_to_hub("SALT-NLP/SMILE", private=True)