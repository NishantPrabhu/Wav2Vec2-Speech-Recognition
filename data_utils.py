
import re
import json
import torch 
import string
import datasets 
import transformers


def remove_punctuation_and_lower(batch):
    ignore_chars = r'[\,\?\.\!\-\;\:\"]'
    batch["text"] = re.sub(ignore_chars, '', batch["text"]).lower()
    return batch

def generate_vocabulary(batch):
    combined = " ".join(batch["text"])
    vocab = list(set(combined))
    return {"vocab": [vocab], "all_text": combined}

def create_vocabulary_file(data, vocab_gen_fn):
    vocab = data.map(vocab_gen_fn, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=data.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0] | set(vocabs["test"]["vocab"][0])))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("./timit_vocab.json", "w") as f:
        json.dump(vocab_dict, f)

def process_timit_dataset():
    data = datasets.load_dataset("timit_asr")
    data = data.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
    data = data.map(remove_punctuation_and_lower)
    


