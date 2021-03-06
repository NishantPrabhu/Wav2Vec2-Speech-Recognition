
import re
import json
import torch 
import string
import librosa
import datasets 
import soundfile
import transformers
import numpy as np


def remove_punctuation_and_lower(texts):
    punctuation = re.sub(r"\'", r"", string.punctuation)
    for i in range(len(texts)):
        texts[i] = texts[i].translate(str.maketrans("", "", punctuation)).upper()
    return texts

def create_vocabulary_file(texts):
    vocab_list = list(set(" ".join(texts)))    
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict.pop(" ")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("./timit_vocab.json", "w") as f:
        json.dump(vocab_dict, f)

def process_timit_dataset(read_limit=2500):
    timit = datasets.load_dataset("timit_asr")
    timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
    train_files, train_text = timit["train"]["file"][:read_limit], timit["train"]["text"][:read_limit]
    test_files, test_text = timit["test"]["file"][:read_limit], timit["test"]["text"][:read_limit]
    train_text = remove_punctuation_and_lower(train_text)
    test_text = remove_punctuation_and_lower(test_text)
    create_vocabulary_file(train_text + test_text)
    return {"file": train_files, "text": train_text}, {"file": test_files, "text": test_text}


class TimitDataloader:

    def __init__(self, data, batch_size):
        self.processor = transformers.Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.files, self.text = data["file"], data["text"]
        self.batch_size = batch_size
        self.ptr = 0

    def __len__(self):
        return len(self.files) // self.batch_size 

    def flow(self):
        speech, text = [], []
        for _ in range(self.batch_size):
            signal, sr = soundfile.read(self.files[self.ptr], dtype="float32")
            speech.append(signal)
            text.append(self.text[self.ptr])
            self.ptr += 1
            if self.ptr >= len(self.files):
                self.ptr = 0

        inputs = self.processor(speech, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
        input_data, input_attention = inputs["input_values"], inputs["attention_mask"]
        with self.processor.as_target_processor():
            labels = self.processor(text, padding=True, return_tensors="pt")
            targets, attention_mask = labels["input_ids"], labels["attention_mask"]
            targets = targets.masked_fill(attention_mask.ne(1), -100)
        return input_data, input_attention, targets

    def generate_from_file(self, file_path):
        signal, sr = soundfile.read(file_path, dtype="float32")
        signal = librosa.resample(np.mean(signal, axis=1), orig_sr=sr, target_sr=16000)
        inputs = self.processor(signal, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
        input_data, input_attention = inputs["input_values"], inputs["attention_mask"]
        return input_data, input_attention


def get_dataloaders(batch_size, read_limit=2500):
    train_data, test_data = process_timit_dataset(read_limit=read_limit)
    train_loader = TimitDataloader(train_data, batch_size)
    test_loader = TimitDataloader(test_data, batch_size)
    return train_loader, test_loader

    
        
if __name__ == "__main__":

    train_data, test_data = process_timit_dataset()
    train_loader = TimitDataloader(train_data, batch_size=4)

    inputs, input_attention, targets = train_loader.flow()
    print(inputs)
    print(input_attention)
    print(targets)