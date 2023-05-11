# imports
from transformers import BertTokenizer
from model import BERTScorer
from data import RetrieverDataset


if __name__ == "__main__":
    dataset = RetrieverDataset("data/truthful_qa")
    for i in range(2):  # We test the first two questions
        question, top_k_spans = dataset[i]
        print(f"Question: {question}")
        for score, span in top_k_spans:
            print(f"Score: {score}, Span: {span}")
        print()
