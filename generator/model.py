from transformers import BertModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer
import torch


class GenerativeQAModel(torch.nn.Module):
    def __init__(self, bert_model_name, gpt2_model_name):
        super().__init__()

        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

    def forward(self, question, spans):
        # Process the spans using BERT model
        span_embs = []
        for span in spans:
            # inputs = self.bert_tokenizer(
            #     span, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.bert_model(span)
            # Use the mean of the span embeddings
            span_embs.append(outputs.last_hidden_state.mean(dim=1))

        span_embs = torch.cat(span_embs)  # Concatenate all span embeddings

        # Generate an answer using GPT-2 model
        question_string = self.bert_tokenizer.decode(question.tolist())
        gpt2_input = self.gpt2_tokenizer.encode(
            question_string, return_tensors="pt")  # Encode the question
        # Concatenate question and span embeddings
        gpt2_input = torch.cat([gpt2_input, span_embs], dim=1)

        output_sequences = self.gpt2_model.generate(
            gpt2_input, max_length=100)  # Generate an answer

        generated_answer = self.gpt2_tokenizer.decode(
            output_sequences[0], skip_special_tokens=True)  # Decode the answer

        return generated_answer
