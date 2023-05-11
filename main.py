# main.py

from retriever.train import train_retriever
from generator.train import train_generator
from retriever.inference import load_retriever_model, retrieve_sentences
from generator.inference import load_generator_model, generate_answer
from transformers import BertTokenizer, GPT2Tokenizer

# Training phase
train_retriever()  # Uncomment this line to train the retriever model
train_generator()  # Uncomment this line to train the generator model

# Inference phase
# Substitute with the path to your retriever model
# retriever_model = load_retriever_model("path/to/retriever_model.pt")
# # Substitute with the path to your generator model
# generator_model = load_generator_model("path/to/generator_model.pt")
# retriever_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# question = "What happens to you if you eat watermelon seeds?"
# document = "Long document text here..."  # Substitute with your document text

# spans = retrieve_sentences(
#     question, document, retriever_model, retriever_tokenizer)
# answer = generate_answer(spans, generator_model, generator_tokenizer)
# print(answer)
