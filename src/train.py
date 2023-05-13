# File: src/train.py

from torch.utils.data import DataLoader
from transformers import AdamW, logging
from bert_score import score
from data import TruthfulQADataset
from datasets import load_dataset, load_metric
from model import Retriever, Generator, RetrieverGenerator
import torch

from sentence_transformers import SentenceTransformer, util
# Instantiate the SentenceTransformer once
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# def calculate_loss(generated_answer, best_answer, correct_answers, incorrect_answers):
#     # calculate Sentence Transformer similarity for the generated answer with respect to best_answer, correct_answers, and incorrect_answers
#     # print('generated_answer: ', generated_answer)
#     # print('best_answer: ', best_answer)
#     # print('correct_answers: ', correct_answers)
#     # print('incorrect_answers: ', incorrect_answers)

#     # Convert to embeddings
#     generated_emb = st_model.encode([generated_answer], convert_to_tensor=True)
#     best_emb = st_model.encode(best_answer, convert_to_tensor=True)

#     # Flatten the list of tuples to a list of strings
#     correct_answers_flat = [
#         item for sublist in correct_answers for item in sublist]
#     incorrect_answers_flat = [
#         item for sublist in incorrect_answers for item in sublist]

#     correct_embs = st_model.encode(
#         correct_answers_flat, convert_to_tensor=True)
#     incorrect_embs = st_model.encode(
#         incorrect_answers_flat, convert_to_tensor=True)

#     # Compute similarities
#     sim_best = util.pytorch_cos_sim(generated_emb, best_emb)
#     sim_correct = torch.mean(torch.stack(
#         [util.pytorch_cos_sim(generated_emb, emb) for emb in correct_embs]))
#     sim_incorrect = torch.mean(torch.stack(
#         [util.pytorch_cos_sim(generated_emb, emb) for emb in incorrect_embs]))

#     # Calculate loss
#     loss = (sim_incorrect - 10 * sim_best - sim_correct)
#     loss = torch.tensor(loss.item(), requires_grad=True)
#     return loss


def calculate_loss(generated_answer, best_answer, correct_answers, incorrect_answers):
    # calculate BERTScore for the generated answer with respect to best_answer, correct_answers, and incorrect_answers
    # print('generated_answer: ', generated_answer)
    # print('best_answer: ', best_answer)
    # print('correct_answers: ', correct_answers)
    # print('incorrect_answers: ', incorrect_answers)
    P_best, _, _ = score([generated_answer], [best_answer],
                         lang='en', model_type="bert-base-uncased")
    P_correct, _, _ = score([generated_answer]*len(correct_answers),
                            correct_answers, lang='en', model_type="bert-base-uncased")
    P_incorrect, _, _ = score([generated_answer]*len(incorrect_answers),
                              incorrect_answers, lang='en', model_type="bert-base-uncased")

    loss = P_incorrect.mean() - 10 * P_best - P_correct.mean()
    loss = torch.tensor(loss.item(), requires_grad=True)
    return loss


def train(model, dataset, batch_size=1, epochs=5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            question_tokens = batch['question']
            spans_tokens = batch['spans']
            best_answer = batch['best_answer']
            correct_answers = batch['correct_answers']
            incorrect_answers = batch['incorrect_answers']

            optimizer.zero_grad()

            # # Feed the question and spans to the Retriever
            # top_k_spans_tokens = model.retriever(question_tokens, spans_tokens)

            # # Feed the question and top k spans to the Generator
            # generated_answers = model.generator(
            #     question_tokens, top_k_spans_tokens)
            generated_answers = model(question_tokens, spans_tokens)

            # Calculate the loss
            loss = calculate_loss(
                generated_answers, best_answer, correct_answers, incorrect_answers)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} Batch {i+1} Loss: {loss.item()}")
                torch.save(model.state_dict(),
                           f"checkpoints/model_{epoch+1}.pth")

        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")


if __name__ == "__main__":
    dataset_path = "data/truthful_qa"
    dataset = TruthfulQADataset(dataset_path)
    model = RetrieverGenerator()
    logging.set_verbosity_error()
    train(model, dataset)
