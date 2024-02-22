import os
import tempfile
from functools import partial

import numpy as np
import ray
import ray.train.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from ftlangdetect import detect
from pyarrow import csv
from ray import train
from sentence_transformers import SentenceTransformer
from torch.optim import Adam
from torch.utils.data import DataLoader

parse_options = csv.ParseOptions(delimiter="\t")


class SentenceEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __call__(self, batch):
        outputs = self.model.encode(
            [sentence for sentence in batch["sentence"]],
            batch_size=len(batch["sentence"]),
        )

        batch["embed"] = [output for output in outputs]
        return batch


def langid(row, key=None):
    fasttext_output = detect(text=row[key].replace("\n", ""), low_memory=False)

    ft_lang = fasttext_output["lang"]
    ft_prob = fasttext_output["score"]
    if ft_lang == "en" and ft_prob >= 0.9:
        return True
    else:
        return False


def get_first_instruction(row):
    sentence = row["conversation"][0]["content"]
    row["sentence"] = sentence
    return {"sentence": sentence}


def apply_label(row, label=None):
    row["label"] = label
    return row


if not os.path.isfile("./processing_steps/embedded/10_000000_000000.parquet"):
    cv = (
        ray.data.read_csv(
            "./common_voice_16_1/transcript/en/train.tsv", parse_options=parse_options
        )
        .repartition(num_blocks=32)
        .filter(partial(langid, key="sentence"))
        .map(partial(apply_label, label=0))
    )

    lc = (
        ray.data.read_parquet("./lmsys-chat-1m/data/")
        .repartition(num_blocks=32)
        .map(get_first_instruction)
        .filter(partial(langid, key="sentence"))
        .map(partial(apply_label, label=1))
    )
    df = cv.union(lc).map_batches(
        SentenceEncoder,
        concurrency=6,
        num_gpus=1,
        batch_size=1024,
    )

    df.write_parquet("./processing_steps/embedded")


class DatasetClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(DatasetClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, encoding):
        return F.log_softmax(self.linear(encoding), dim=1)


df = (
    ray.data.read_parquet("/data/wheld3/audio/accents/processing_steps/embedded")
    .repartition(num_blocks=4)
    .materialize()
)


if not os.path.isfile("./model.pt"):
    df = df.random_shuffle(seed=42)

    def train_func(config):
        # Model, Loss, Optimizer
        model = DatasetClassifier(2, 384)

        # [1] Prepare model.
        model = ray.train.torch.prepare_model(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        # Data
        train_ds = train.get_dataset_shard("train")

        # Training
        for epoch in range(4):
            for batch in train_ds.iter_batches(batch_size=128):
                embed = torch.tensor(np.stack(batch["embed"].tolist())).to("cuda")
                label = torch.tensor(batch["label"]).to("cuda")
                outputs = model(embed)
                loss = criterion(outputs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Report metrics and checkpoint.
            metrics = {"loss": loss.item(), "epoch": epoch}
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
                )
                ray.train.report(
                    metrics,
                    checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
                )
            if ray.train.get_context().get_world_rank() == 0:
                print(metrics)

    # [4] Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=True)

    # [5] Launch distributed training job.
    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        datasets={"train": df},
        #    dataset_config=train.DataConfig(
        #        datasets_to_split=["train"],
        #    ),
    )
    result = trainer.fit()


class Classifier:
    def __init__(self):
        model_state_dict = torch.load("./model.pt", map_location="cpu")
        model = DatasetClassifier(2, 384).to("cpu")
        model.load_state_dict(model_state_dict)
        self.model = model

    @torch.no_grad
    def __call__(self, batch):
        outputs = self.model(torch.tensor(np.stack(batch["embed"].tolist())))
        batch["pred"] = [F.softmax(output, dim=0).numpy()[1] for output in outputs]
        return batch


df.map_batches(
    Classifier,
    concurrency=32,
    num_gpus=0,
    batch_size=64,
).write_parquet("./processing_steps/labeled")
