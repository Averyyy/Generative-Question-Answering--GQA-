# retriever/data.py

from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import BertTokenizer
from model import get_top_k_spans


class RetrieverDataset(Dataset):
    def __init__(self, path, model_name='bert-base-uncased', k=5):
        # Load the validation set
        self.data = load_from_disk(path)['validation']
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.k = k

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        source_text = item['source_text']
        top_k_spans = get_top_k_spans(question, source_text, self.k)
        return question, top_k_spans
