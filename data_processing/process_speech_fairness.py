from datasets import Dataset
import pandas as pd
from datasets import Audio
import os

df = pd.read_csv("/Users/wyshi/Downloads/speech_fairness.txt", sep="\t")

PREFIX = "/Users/wyshi/Downloads/asr_fairness_audio/"

df["audio"] = (PREFIX + df["hash_name"] + ".wav").tolist()

# convert NA
df = df.where(pd.notna(df), None)
audio_dataset = Dataset.from_dict(
    df[
        [
            "audio",
            "transcription",
            "age",
            "gender",
            "first_language",
            "socioeconomic_bkgd",
            "ethnicity",
        ]
    ].to_dict(orient="list")
).cast_column("audio", Audio())

audio_dataset.push_to_hub("SALT-NLP/speech_fairness", private=True)
