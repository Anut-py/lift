from model.embedding import gen_embeddings
import torch
tensor = torch.load("data-files/embeddings.pt")
with open("data-files/facts.txt", "r", encoding='utf-8') as f:
  facts = f.readlines()
torch.save(gen_embeddings(facts), "data-files/embeddings.pt")