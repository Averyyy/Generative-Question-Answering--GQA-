# File: src/model.py

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
from torch import nn


class Retriever(nn.Module):
    def __init__(self):
        super(Retriever, self).__init__()
        self.bert = BertModel(BertConfig())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, question_tokens, spans_tokens):
        # We calculate the relevance score for each span
        scores = []
        for span_token in spans_tokens:
            # Concatenate question and span tokens
            span_token = list(span_token)
            input_tokens = ['[CLS]'] + question_tokens + \
                ['[SEP]'] + span_token + ['[SEP]']
            input_ids = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(input_tokens)])
            attention_mask = torch.tensor([[1] * len(input_tokens)])

            # Get BERT output
            output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask)

            # Calculate relevance score as the dot product between question and span CLS token embeddings
            question_cls_embedding = output['last_hidden_state'][0][0]
            span_cls_embedding = output['last_hidden_state'][0][len(
                question_tokens) + 2]
            score = torch.dot(question_cls_embedding, span_cls_embedding)
            scores.append(score)

        # Get the top k spans based on the relevance scores
  # Get the top k spans based on the relevance scores
        k = 5
        top_k_scores, top_k_indices = torch.topk(
            torch.tensor(scores), min(k, len(scores)))
        top_k_spans_tokens = [spans_tokens[i] for i in top_k_indices]

        return top_k_spans_tokens


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def forward(self, question_tokens, top_k_spans_tokens):
        # Concatenate question and all span tokens together
        input_tokens = ['[CLS]']  # + question_tokens + ['[SEP]']
        for span_token in top_k_spans_tokens:
            span_token = list(span_token)
            input_tokens += span_token + ['[SEP]']
        input_tokens += question_tokens + ['[SEP]']

        # Convert input tokens to ids
        input_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(input_tokens)])

        # Generate an answer using the concatenated tokens
        output = self.gpt2.generate(
            input_ids, pad_token_id=self.tokenizer.eos_token_id)
        generated_answer = self.tokenizer.decode(output[0])

        return generated_answer


class RetrieverGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.retriever = Retriever()
        self.generator = Generator()

    def forward(self, question_tokens, spans_tokens):
        top_k_spans_tokens = self.retriever(question_tokens, spans_tokens)
        generated_answer = self.generator(question_tokens, top_k_spans_tokens)
        return generated_answer
