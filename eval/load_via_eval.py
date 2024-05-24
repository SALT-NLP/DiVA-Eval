from datasets import load_dataset

SEED = 42
TEST_SIZE = 0.2  # if no val/test already


def load_via_eval(
    dataset_name,
):
    if dataset_name == "WillHeld/SD-QA":
        return load_SDQA(dataset_name)
    elif dataset_name == "SALT-NLP/speech_fairness":
        return load_speech_fairness(dataset_name)


def load_SDQA(dataset_name="WillHeld/SD-QA"):
    # https://huggingface.co/datasets/WillHeld/SD-QA
    # the name of x and y
    x_label, y_label = "question", "answers"
    # load the right partition
    ds = load_dataset(dataset_name)["dev"]
    # filter
    ds = ds.filter(lambda example: example[y_label])

    return x_label, y_label, ds


def load_speech_fairness(dataset_name="SALT-NLP/speech_fairness"):
    # https://huggingface.co/datasets/SALT-NLP/speech_fairness
    # the name of x and y
    x_label, y_label = "audio", "transcription"
    # load the right partition
    ds = load_dataset(dataset_name).train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds


def load_HeySquad(dataset_name="yijingwu/HeySQuAD_human"):
    # https://huggingface.co/datasets/yijingwu/HeySQuAD_human
    # the name of x and y
    x_label, y_label = "audio", "answers"
    # load the right partition
    ds = load_dataset(dataset_name)["validation"]
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds
