# File: src/inference.py
from model import RetrieverGenerator
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import re

class InferenceDataset(Dataset):
    def __init__(self, question, long_text):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = self.tokenizer.model_max_length
        self.question = question
        self.long_text = long_text

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        question_tokens = self.tokenizer.tokenize(self.question)

        # Sliding window approach to extract spans from the long_text
        spans = []
        long_text_words = re.split('; |, |\n|\t| ', self.long_text)
        for i in range(0, len(long_text_words), self.max_length - len(question_tokens) - 3):
            span_words = long_text_words[i:i + self.max_length - len(question_tokens) - 3]
            span = ' '.join(span_words)
            spans.append(span)

        return {'question': question_tokens, 'spans': spans}

def infer(question, long_text):
    model = RetrieverGenerator()
    # Load the most recent model
    model.load_state_dict(torch.load('checkpoints/model_5.pth'))

    # Create an inference dataset
    dataset = InferenceDataset(question, long_text)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            question_tokens = batch['question']
            spans_tokens = batch['spans']
            generated_answer = model(question_tokens, spans_tokens)
            return generated_answer

if __name__ == "__main__":
    question = "Who won the world series in 2020?"
    long_text = """
    The Los Angeles Dodgers won the National League pennant and faced the American League champion Tampa Bay Rays in the Series. 
    The Dodgers defeated the Rays to win their first World Series in 32 years; the Rays had been seeking their first championship, having lost their previous Series appearance in 2008.
    """
    print(infer(question, long_text))
