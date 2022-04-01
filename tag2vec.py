from konlpy.tag import Twitter ; tw = Twitter()
from gensim.models import Word2Vec
import torch
import torch.nn as nn


#gensim model created
model = Word2Vec(reviews,size=100, window=5, min_count=5, workers=4)


weights = torch.FloatTensor(model.wv.vectors)
embedding = nn.Embedding.from_pretrained(weights)

