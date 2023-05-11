# model.py

from transformers import BertTokenizer
from transformers import BertModel, BertTokenizer
import torch


class BERTScorer:
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    def score(self, question, span):
        # We assume that both question and span are strings
        with torch.no_grad():
            question_input = self.tokenizer(
                question, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            span_input = self.tokenizer(
                span, return_tensors='pt', truncation=True, max_length=512).to(self.device)

            question_outputs = self.model(**question_input)
            span_outputs = self.model(**span_input)

            question_embedding = question_outputs.last_hidden_state.mean(dim=1)
            span_embedding = span_outputs.last_hidden_state.mean(dim=1)

            # Cosine similarity as a measure of relevance
            relevance = torch.nn.functional.cosine_similarity(
                question_embedding, span_embedding)
        return relevance


# def get_top_k_spans(question, source_text, k, span_len=512):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     scorer = BERTScorer()

#     # Break down the source text into spans
#     tokens = tokenizer.tokenize(source_text)
#     spans = [tokens[i: i+span_len] for i in range(0, len(tokens), span_len)]

#     # Score each span
#     span_scores = []
#     for span in spans:
#         span_text = tokenizer.convert_tokens_to_string(span)
#         score = scorer.score(question, span_text)
#         span_scores.append((score, span_text))

#     # Sort spans by their scores and return the top k spans
#     top_k_spans = sorted(span_scores, key=lambda x: x[0], reverse=True)[:k]

#     return top_k_spans


def get_top_k_spans(question_ids, source_text_ids, k=5, span_len=500):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    scorer = BERTScorer()

    # Initialize lists to store results for all examples in the batch
    batch_top_k_spans = []

    # Loop over each example in the batch
    for i in range(question_ids.size(0)):
        # Decode the tokenized input ids to text
        question = tokenizer.decode(question_ids[i])
        source_text = tokenizer.decode(source_text_ids[i])

        # Break down the source text into spans
        tokens = tokenizer.tokenize(source_text)
        spans = [tokens[j: j+span_len]
                 for j in range(0, len(tokens), span_len)]

        # Score each span
        span_scores = []
        for span in spans:
            span_text = tokenizer.convert_tokens_to_string(span)
            score = scorer.score(question, span_text)
            span_scores.append((score, span_text))

        # Sort spans by their scores and return the top k spans
        top_k_spans_text = sorted(
            span_scores, key=lambda x: x[0], reverse=True)[:k]

        # Tokenize top-k spans
        top_k_spans_ids = [tokenizer.encode(
            span[1], add_special_tokens=False) for span in top_k_spans_text]

        # Pad top-k spans to ensure they all have the same length
        top_k_spans_ids_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(span) for span in top_k_spans_ids], batch_first=True)

        batch_top_k_spans.append(top_k_spans_ids_padded)

    return batch_top_k_spans
