import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
from constants import *
import torch
from dataloader import get_dataloader

class VAEModel(pl.LightningModule):

    def __init__(self,id2word_path='data/word2id.json',latent_size=32):
        super(VAEModel, self).__init__()
        with open(id2word_path,'r') as f:
           id2word=json.load(f)
        self.latent_size = latent_size
        self.enc_word_embed = nn.Embedding(len(id2word),EMBED_SIZE,padding_idx=0)
        self.enc_lstm = nn.LSTM(EMBED_SIZE,EMBED_SIZE,batch_first=True)
        self.hidden2mean = nn.Linear(EMBED_SIZE,latent_size)
        self.hidden2logv = nn.Linear(EMBED_SIZE,latent_size)
        self.latent2hidden = nn.Linear(latent_size,EMBED_SIZE)
        self.dec_lstm  = nn.Linear(EMBED_SIZE,300)
        self.hidden2word = nn.Linear(EMBED_SIZE,len(id2word))

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        inp, inp_len = batch[0],batch[1]
        batch_size = inp.size(0)
        word_embeds = self.enc_word_embed(inp)
        word_embeds = pack_padded_sequence(word_embeds,inp_len,True,False)
        hidden, (hn,cn) = self.enc_lstm(word_embeds)

        mean = self.hidden2mean(hn)
        logv = self.hidden2logv(hn)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size,self.latent_size], requires_grad=True)
        new_z = z*mean + std

        dec_z = self.latent2hidden(z)
        out,_  = self.dec_lstm(word_embeds,dec_z)

        out = pad_packed_sequence(out,True)[0]
        out = out.contiguous()

        logits = self.hidden2word(out)
        loss1 = nn.functional.cross_entropy(logits, inp)
        loss2 = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        loss3 = nn.functional.kl_div(new_z, z)
        loss = loss1 + loss2
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    obj = VAEModel()
    pl_trainer = pl.Trainer()
    pl_trainer.fit(obj,get_dataloader('data/train.csv'))