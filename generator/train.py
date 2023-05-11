from transformers import BertModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer
from datasets import load_from_disk
from torch.optim import Adam
from transformers import BertTokenizer, BertModel
from bert_score import score as bert_score
from torch.utils.data import DataLoader
from data import GenerativeQADataset
from model import GenerativeQAModel
from retriever.model import get_top_k_spans
import torch
from torch.nn.utils.rnn import pad_sequence

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_fn(batch):
    questions, best_answers, correct_answers, incorrect_answers, source_texts = zip(
        *batch)

    # tokenize and convert to tensors
    questions = [torch.tensor(tokenizer.encode(q)) for q in questions]
    best_answers = [torch.tensor(tokenizer.encode(ba)) for ba in best_answers]
    source_texts = [torch.tensor(tokenizer.encode(st)) for st in source_texts]

    # pad sequences
    questions = pad_sequence(questions, batch_first=True)
    best_answers = pad_sequence(best_answers, batch_first=True)
    source_texts = pad_sequence(source_texts, batch_first=True)

    # handle correct_answers and incorrect_answers separately because they are nested lists
    correct_answers = [pad_sequence([torch.tensor(tokenizer.encode(
        ca)) for ca in cas], batch_first=True) for cas in correct_answers]
    incorrect_answers = [pad_sequence([torch.tensor(tokenizer.encode(
        ia)) for ia in ias], batch_first=True) for ias in incorrect_answers]

    return questions, best_answers, correct_answers, incorrect_answers, source_texts


def train(model, dataset, bert_model_name, batch_size=1, epochs=10, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):

        for i, row in enumerate(dataloader):
            question, best_answer, correct_answers, incorrect_answers, source_text = row
            optimizer.zero_grad()

            # Process the source text into top k spans
            spans = get_top_k_spans(question, source_text)

            # Feed the question and spans into the model to generate an answer
            generated_answer = model(question, spans)

            # Calculate the BertScore loss
            P = bert_score(generated_answer, [best_answer] + correct_answers + incorrect_answers,
                           lang='en', model_type=bert_model_name, rescale_with_baseline=True)

            correct_scores = P[:len(correct_answers)+1].mean()
            incorrect_scores = P[len(correct_answers)+1:].mean()

            loss = incorrect_scores - 5*correct_scores

            # Backpropagation
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(
                    f'Epoch {epoch+1}/{epochs}, step {i+1}/{len(dataloader)}, loss {loss.item()}')

        print(f'Epoch {epoch+1}/{epochs} finished.')


if __name__ == '__main__':
    from transformers import BertModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer
    from datasets import load_from_disk
    loaded_dataset = load_from_disk("data/truthful_qa")
    dataset = GenerativeQADataset(loaded_dataset)

    bert_model_name = 'bert-base-uncased'
    gpt2_model_name = 'gpt2'

    model = GenerativeQAModel(bert_model_name, gpt2_model_name)

    train(model, dataset, bert_model_name, batch_size=32, epochs=10, lr=0.001)
