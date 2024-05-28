from datasets import load_dataset

SEED = 42
TEST_SIZE = 0.2  # if no val/test already


def load_via_eval(
    dataset_name,
    language=None,
):
    if dataset_name == "Spoken_Dialect_QA":
        return load_SDQA(language=language)
    elif dataset_name == "speech_fairness":
        return load_speech_fairness()
    elif dataset_name == "non_social_HeySquad_QA":
        return load_HeySquad()
    elif dataset_name == "non_social_SLURP_speaker_intent":
        return load_SLURP()
    elif dataset_name == "IEMOCAP_emotion_recognition":
        return load_IEMOCAP_emotion_recognition()
    elif dataset_name == "MELD_emotion_recognition":
        return load_MELD_emotion_recognition()
    elif dataset_name == "Mustard_sarcasm":
        return load_mustard_sarcasm()
    elif dataset_name == "CommonVoice_speaker_identity":
        return load_commonvoice_classification(language=language)
    elif dataset_name == "FLEURS_speaker_identity":
        return load_google_fleurs_speaker_identify(language=langauge)
    elif dataset_name == "Callhome_relationships":
        return load_callhome_relationships()
    elif dataset_name == "URFunny_humor":
        return load_urfunny_humor()


def load_SDQA(language, dataset_name="WillHeld/SD-QA"):
    # https://huggingface.co/datasets/WillHeld/SD-QA
    # the name of x and y
    x_label, y_label = language, "answers"
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
    def extract_answer(ex):
        ex["answers"] = [answer_json["text"] for answer_json in ex["answers"]]
        return ex

    # https://huggingface.co/datasets/yijingwu/HeySQuAD_human
    # the name of x and y
    x_label, y_label = "audio", "answers"
    # load the right partition
    ds = load_dataset(dataset_name)["validation"]  # no test partition
    # filter
    ds = ds.filter(lambda example: example[y_label]).map(extract_answer)
    return x_label, y_label, ds


def load_SLURP(dataset_name="qmeeus/slurp"):
    # https://huggingface.co/datasets/qmeeus/slurp
    # the name of x and y
    x_label, y_label = "audio", "intent"
    # load the right partition
    ds = load_dataset(dataset_name)["test"]
    # filter
    ds = ds.filter(lambda example: example[y_label])

    label_dict = {
        "0": "addcontact",
        "1": "alarm_query",
        "2": "alarm_remove",
        "3": "alarm_set",
        "4": "audio_volume_down",
        "5": "audio_volume_mute",
        "6": "audio_volume_other",
        "7": "audio_volume_up",
        "8": "calendar_query",
        "9": "calendar_remove",
        "10": "calendar_set",
        "11": "cleaning",
        "12": "coffee",
        "13": "convert",
        "14": "cooking_query",
        "15": "cooking_recipe",
        "16": "createoradd",
        "17": "currency",
        "18": "datetime_convert",
        "19": "datetime_query",
        "20": "definition",
        "21": "email_addcontact",
        "22": "email_query",
        "23": "email_querycontact",
        "24": "email_sendemail",
        "25": "events",
        "26": "factoid",
        "27": "game",
        "28": "general_affirm",
        "29": "general_commandstop",
        "30": "general_confirm",
        "31": "general_dontcare",
        "32": "general_explain",
        "33": "general_greet",
        "34": "general_joke",
        "35": "general_negate",
        "36": "general_praise",
        "37": "general_quirky",
        "38": "general_repeat",
        "39": "greet",
        "40": "hue_lightdim",
        "41": "hue_lightoff",
        "42": "hue_lightup",
        "43": "iot_cleaning",
        "44": "iot_coffee",
        "45": "iot_hue_lightchange",
        "46": "iot_hue_lightdim",
        "47": "iot_hue_lightoff",
        "48": "iot_hue_lighton",
        "49": "iot_hue_lightup",
        "50": "iot_wemo_off",
        "51": "iot_wemo_on",
        "52": "joke",
        "53": "likeness",
        "54": "lists_createoradd",
        "55": "lists_query",
        "56": "lists_remove",
        "57": "locations",
        "58": "music",
        "59": "music_dislikeness",
        "60": "music_likeness",
        "61": "music_query",
        "62": "music_settings",
        "63": "news_query",
        "64": "play_audiobook",
        "65": "play_game",
        "66": "play_music",
        "67": "play_podcasts",
        "68": "play_radio",
        "69": "podcasts",
        "70": "post",
        "71": "qa_currency",
        "72": "qa_definition",
        "73": "qa_factoid",
        "74": "qa_maths",
        "75": "qa_stock",
        "76": "query",
        "77": "querycontact",
        "78": "quirky",
        "79": "radio",
        "80": "recommendation_events",
        "81": "recommendation_locations",
        "82": "recommendation_movies",
        "83": "remove",
        "84": "sendemail",
        "85": "set",
        "86": "settings",
        "87": "social_post",
        "88": "social_query",
        "89": "takeaway_order",
        "90": "takeaway_query",
        "91": "ticket",
        "92": "traffic",
        "93": "transport_query",
        "94": "transport_taxi",
        "95": "transport_ticket",
        "96": "transport_traffic",
        "97": "volume_other",
        "98": "weather_query",
        "99": "wemo_off",
        "100": "wemo_on",
    }

    def map_to_label(example):
        example["text_intent"] = label_dict[str(example["intent"])]
        return example

    updated_ds = ds.map(map_to_label)
    return x_label, "text_intent", updated_ds


def load_IEMOCAP_emotion_recognition(
    dataset_name="Zahra99/IEMOCAP_Audio",
):
    # https://huggingface.co/datasets/Zahra99/IEMOCAP_Audio
    # the name of x and y
    x_label, y_label = "audio", "label"
    # load the right partition
    ds = load_dataset(dataset_name)  # there are multiple sessions
    # filter
    ds = ds.filter(lambda example: example[y_label])

    label_dict = {"0": "angry", "1": "happy", "2": "neutral", "3": "sad"}

    def map_to_label(example):
        example["emotion_label"] = label_dict[str(example[y_label])]
        return example

    updated_ds = ds.map(map_to_label)
    return x_label, "emotion_label", updated_ds


def load_MELD_emotion_recognition(
    dataset_name="DavidCombei/Wav2Vec_MELD_Audio",
):
    # https://huggingface.co/datasets/DavidCombei/Wav2Vec_MELD_Audio
    # the name of x and y
    x_label, y_label = "audio", "label"
    # load the right partition
    ds = load_dataset(dataset_name)["test"]  # there are multiple sessions
    # filter
    ds = ds.filter(lambda example: example[y_label])

    label_dict = {
        "0": "anger",
        "1": "disgust",
        "2": "fear",
        "3": "joy",
        "4": "neutral",
        "5": "sadness",
        "6": "surprise",
    }

    def map_to_label(example):
        example["emotion_label"] = label_dict[str(example[y_label])]
        return example

    updated_ds = ds.map(map_to_label)
    return x_label, "emotion_label", updated_ds


def load_mustard_sarcasm(dataset_name="SALT-NLP/Mustard_sarcasm"):
    # website, e.g., https://huggingface.co/datasets/yijingwu/HeySQuAD_human
    # the name of x and y
    x_label, y_label = "audio", "sarcasm"
    # load the right partition
    ds = load_dataset(dataset_name).train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )  # no test partition
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds


def load_commonvoice_classification(
    language,
    dataset_name="mozilla-foundation/common_voice_17_0",
):
    import numpy as np

    # https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    # features = ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']
    # example_data = {
    #     "client_id": "000abb3006b78ea4c1144e55d9d158f05a9db0110160510fef2b006f2c2c8e35f7bb538b04542511834b61503cdda5b0331566a5cf59dc0d375a44afc4d10777",
    #     "path": "en_test_0/common_voice_en_27710027.mp3",
    #     "audio": {
    #         "path": "en_test_0/common_voice_en_27710027.mp3",
    #         "array": np.array(
    #             [
    #                 -5.68434189e-13,
    #                 -1.25055521e-12,
    #                 -6.13908924e-12,
    #                 -1.00076851e-03,
    #                 -4.13570320e-04,
    #                 -2.24993564e-05,
    #             ]
    #         ),
    #         "sampling_rate": 48000,
    #     },
    #     "sentence": "Joe Keaton disapproved of films, and Buster also had reservations about the medium.",
    #     "up_votes": 3,
    #     "down_votes": 1,
    #     "age": "",
    #     "gender": "",
    #     "accent": "",
    #     "locale": "en",
    #     "segment": "",
    #     "variant": "",
    # }
    language_list = [
        "ab",
        "af",
        "am",
        "ar",
        "as",
        "ast",
        "az",
        "ba",
        "bas",
        "be",
        "bg",
        "bn",
        "br",
        "ca",
        "ckb",
        "cnh",
        "cs",
        "cv",
        "cy",
        "da",
        "de",
        "dv",
        "dyu",
        "el",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "fy-NL",
        "ga-IE",
        "gl",
        "gn",
        "ha",
        "he",
        "hi",
        "hsb",
        "ht",
        "hu",
        "hy-AM",
        "ia",
        "id",
        "ig",
        "is",
        "it",
        "ja",
        "ka",
        "kab",
        "kk",
        "kmr",
        "ko",
        "ky",
        "lg",
        "lij",
        "lo",
        "lt",
        "ltg",
        "lv",
        "mdf",
        "mhr",
        "mk",
        "ml",
        "mn",
        "mr",
        "mrj",
        "mt",
        "myv",
        "nan-tw",
        "ne-NP",
        "nhi",
        "nl",
        "nn-NO",
        "nso",
        "oc",
        "or",
        "os",
        "pa-IN",
        "pl",
        "ps",
        "pt",
        "quy",
        "rm-sursilv",
        "rm-vallader",
        "ro",
        "ru",
        "rw",
        "sah",
        "sat",
        "sc",
        "sk",
        "skr",
        "sl",
        "sq",
        "sr",
        "sv-SE",
        "sw",
        "ta",
        "te",
        "th",
        "ti",
        "tig",
        "tk",
        "tok",
        "tr",
        "tt",
        "tw",
        "ug",
        "uk",
        "ur",
        "uz",
        "vi",
        "vot",
        "yi",
        "yo",
        "yue",
        "zgh",
        "zh-CN",
        "zh-HK",
        "zh-TW",
        "zu",
        "zza",
    ]
    # the name of x and y
    x_label, y_label = "audio", "sentence"
    ds = load_dataset(dataset_name, language, split="test", streaming=True)
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds


def load_google_fleurs_speaker_identify(language, dataset_name="google/fleurs"):
    # https://huggingface.co/datasets/google/fleurs
    # the name of x and y
    language_list = [
        "af_za",
        "am_et",
        "ar_eg",
        "as_in",
        "ast_es",
        "az_az",
        "be_by",
        "bg_bg",
        "bn_in",
        "bs_ba",
        "ca_es",
        "ceb_ph",
        "ckb_iq",
        "cmn_hans_cn",
        "cs_cz",
        "cy_gb",
        "da_dk",
        "de_de",
        "el_gr",
        "en_us",
        "es_419",
        "et_ee",
        "fa_ir",
        "ff_sn",
        "fi_fi",
        "fil_ph",
        "fr_fr",
        "ga_ie",
        "gl_es",
        "gu_in",
        "ha_ng",
        "he_il",
        "hi_in",
        "hr_hr",
        "hu_hu",
        "hy_am",
        "id_id",
        "ig_ng",
        "is_is",
        "it_it",
        "ja_jp",
        "jv_id",
        "ka_ge",
        "kam_ke",
        "kea_cv",
        "kk_kz",
        "km_kh",
        "kn_in",
        "ko_kr",
        "ky_kg",
        "lb_lu",
        "lg_ug",
        "ln_cd",
        "lo_la",
        "lt_lt",
        "luo_ke",
        "lv_lv",
        "mi_nz",
        "mk_mk",
        "ml_in",
        "mn_mn",
        "mr_in",
        "ms_my",
        "mt_mt",
        "my_mm",
        "nb_no",
        "ne_np",
        "nl_nl",
        "nso_za",
        "ny_mw",
        "oc_fr",
        "om_et",
        "or_in",
        "pa_in",
        "pl_pl",
        "ps_af",
        "pt_br",
        "ro_ro",
        "ru_ru",
        "sd_in",
        "sk_sk",
        "sl_si",
        "sn_zw",
        "so_so",
        "sr_rs",
        "sv_se",
        "sw_ke",
        "ta_in",
        "te_in",
        "tg_tj",
        "th_th",
        "tr_tr",
        "uk_ua",
        "umb_ao",
        "ur_pk",
        "uz_uz",
        "vi_vn",
        "wo_sn",
        "xh_za",
        "yo_ng",
        "yue_hant_hk",
        "zu_za",
        "all",
    ]
    x_label, y_label = "audio", "gender"
    # load the right partition
    ds = load_dataset(
        dataset_name, language, split="test", streaming=True
    )  # no test partition
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds

def load_callhome_relationships(dataset_name="SALT-NLP/Callhome_relationships"):
    # website, e.g., https://huggingface.co/datasets/yijingwu/HeySQuAD_human
    # the name of x and y
    x_label, y_label = "audio", "Speaker A - Primary"
    # load the right partition
    ds = load_dataset(dataset_name).train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )  # no test partition
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds

def load_urfunny_humor(dataset_name="SALT-NLP/URFunny_humor"):
    # website, e.g., https://huggingface.co/datasets/yijingwu/HeySQuAD_human
    # the name of x and y
    x_label, y_label = "audio", "label"
    # load the right partition
    ds = load_dataset(dataset_name, split="test", streaming=True)
    # filter
    ds = ds.filter(lambda example: example[y_label])
    return x_label, y_label, ds


###### this is a template ######
# def load_xxxname(dataset_name="xxxx"):
#     # website, e.g., https://huggingface.co/datasets/yijingwu/HeySQuAD_human
#     # the name of x and y
#     x_label, y_label = xxx, xxx
#     # load the right partition
#     ds = load_dataset(dataset_name)[xxx]  # no test partition
#     # filter
#     ds = ds.filter(lambda example: example[y_label])
#     return x_label, y_label, ds

if __name__ == "__main__":
    x_label, y_label, ds = load_via_eval(
        dataset_name="CommonVoice_speaker_identity",
        language="af_za",
    )
    import pdb

    pdb.set_trace()
