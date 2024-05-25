from moviepy.editor import VideoFileClip
from glob import glob
from tqdm import tqdm

from datasets import Dataset
import pandas as pd
from datasets import Audio


# Function to extract audio
def extract_audio(mp4_file, output_audio_file):
    video = VideoFileClip(mp4_file)
    audio = video.audio
    audio.write_audiofile(
        output_audio_file,
    )


def get_wav():
    for mp4_dir in tqdm(
        glob("/Users/wyshi/Downloads/mmsd_raw_data/utterances_final/*.mp4")
    ):
        extract_audio(
            mp4_dir,
            mp4_dir.replace("utterances_final", "wav_final").replace("mp4", "wav"),
        )


import json

with open("/Users/wyshi/Downloads/mmsd_raw_data/sarcasm_data.json") as fh:
    meta_data = json.load(fh)

data_dict = {
    "audio": [
        f"/Users/wyshi/Downloads/mmsd_raw_data/wav_final/{key}.wav"
        for key in sorted(meta_data.keys())
    ],
    "utterance": [meta_data[key]["utterance"] for key in sorted(meta_data.keys())],
    "speaker": [meta_data[key]["speaker"] for key in sorted(meta_data.keys())],
    "context": [meta_data[key]["context"] for key in sorted(meta_data.keys())],
    "context_speakers": [
        meta_data[key]["context_speakers"] for key in sorted(meta_data.keys())
    ],
    "show": [meta_data[key]["show"] for key in sorted(meta_data.keys())],
    "sarcasm": [meta_data[key]["sarcasm"] for key in sorted(meta_data.keys())],
}
audio_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())

audio_dataset.push_to_hub("SALT-NLP/Mustard_sarcasm", private=True)
