# generator/inference.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .model import GeneratorModel
import torch


def load_generator_model(path_to_model):
    model = GeneratorModel()
    model.load_state_dict(torch.load(path_to_model))
    model.to('cuda')
    model.eval()
    return model


def generate_answer(spans, model, tokenizer, max_length=512):
    text = " ".join(spans)
    encoded = tokenizer.encode(text, return_tensors='pt')
    input_ids = encoded.to('cuda')
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    model = load_generator_model("checkpoints/generator_model.pt")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    spans = ["Span 1", "Span 2", "Span 3", "Span 4",
             "Span 5"]  # Retrieved from retriever model
    answer = generate_answer(spans, model, tokenizer)
    print(answer)
