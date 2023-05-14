#data.py
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import BertTokenizer
import re

class TruthfulQADataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)['validation']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = self.tokenizer.model_max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question = row['question']
        source_text = row['source_text']
        best_answer = row['best_answer']
        correct_answers = row['correct_answers']
        incorrect_answers = row['incorrect_answers']

        # Tokenize the question
        question_tokens = self.tokenizer.tokenize(question)

        # Sliding window approach to extract spans from the source_text
        spans = []
        # source_text_words = source_text.split(' ')
        source_text_words = re.split('; |, |\n|\t| ', source_text)
        for i in range(0, len(source_text_words), self.max_length - len(question_tokens) - 3):
            span_words = source_text_words[i:i +
                                           self.max_length - len(question_tokens) - 3]
            span = ' '.join(span_words)
            spans.append(span)

        return {'question': question,
                'spans': spans,
                'best_answer': best_answer,
                'correct_answers': correct_answers,
                'incorrect_answers': incorrect_answers}