import torch
import pandas as pd
from constants import *
import spacy
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):

    def __init__(self,X):
        super(CustomDataset, self).__init__()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]

def get_vocab(text : list):
    word2id = {}
    ind = 1
    word_vectors = []
    nlp = spacy.load("en_core_web_lg")
    docs = nlp.pipe(text,batch_size=64,disable=["ner","tagger","parser"])
    for doc in docs:
        for token in doc:
            if token.text not in word2id:
                word2id[token.text] = ind
                word_vectors.append(token.vector)
                ind+=1

    with open('data/word2id.json','w') as f:
        json.dump(word2id,f)
    np.save('data/word_embs.npy',np.asarray(word_vectors))
    return word2id, word_vectors

def vectorize_text(text : list, word2id : dict):
    nlp = spacy.load("en_core_web_lg")
    docs = nlp.pipe(text, batch_size=64, disable=["ner", "tagger","parser"])
    X = []
    for ind,doc in enumerate(docs):
        print(ind)
        X.append([word2id[START_TOKEN]])
        for token in doc:
            X[-1].append(word2id[token.text])
        X[-1].append(word2id[END_TOKEN])
    return X

def pad_collate(X):
    X_len = [len(X[i]) for i in range(len(X))]
    X = pad_sequence([torch.LongTensor(X[i]) for i in range(len(X))],batch_first=True)
    return torch.LongTensor(X),torch.LongTensor(X_len)

def get_dataloader(train_path : str, mode=TRAIN_MODE):
    df = pd.read_csv(train_path)[:20000]
    id2question = {}
    for ind,row in enumerate(df.iterrows()):
        print(ind)
        try:
            id2question[int(row[1]['qid1'])] = row[1]['question1'].lower()
            id2question[int(row[1]['qid2'])] = row[1]['question2'].lower()
        except Exception as e:
            print(e)
    questions = list(id2question.values())
    get_vocab(questions)
    with open('data/word2id.json') as f:
        word2id = json.load(f)

    word2id[PAD_TOKEN]= 0
    word2id[START_TOKEN] = len(word2id)
    word2id[END_TOKEN] = len(word2id)

    X = vectorize_text(questions, word2id)
    ds = CustomDataset(X)
    dataloader = DataLoader(ds,collate_fn=pad_collate,batch_size=64)

    return dataloader


if __name__ == "__main__":
    dl = get_dataloader('data/train.csv')
    for batch in dl:
        print(batch)