
import torch 
import datasets 
import transformers


def process_timit_dataset():
    data = datasets.load_dataset("timit_asr")