# File: src/retriever/model.py
from transformers import BertModel, BertTokenizer
from bert_score import score
import torch


def get_top_k_spans(question, source_text, k=5, max_length=512):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Split the text into spans based on sentences
    spans = source_text.split('. ')
    spans = [span for span in spans if len(span) > 0]

    # Score each span
    scores = []
    for span in spans:
        inputs = tokenizer.encode_plus(question, span,
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       padding='max_length',
                                       truncation=True,
                                       return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            scores.append(cls_embedding)

    # Get the top k spans
    scores, indices = torch.topk(torch.Tensor(scores), k)
    top_k_spans = [spans[i] for i in indices]

    return top_k_spans
