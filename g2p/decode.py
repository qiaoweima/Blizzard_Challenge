import torch
import numpy as np
from utils import *
from torch.nn.utils.rnn import pad_sequence
import speechbrain as sb
from speechbrain.lobes.models.g2p.model import TransformerG2P
from transformers import FlaubertModel, FlaubertTokenizer
import torch.nn as nn
from speechbrain.dataio import dataset, dataloader
import torch
import torch.nn.functional as F
import ptvsd

# ptvsd.enable_attach(("localhost",5601))
# ptvsd.wait_for_attach()
test_text = "électroraffinâmes"

phn_len = len(symbols)
grapheme_len = len(graphemes)
emb_dim = 32
d_model = 256

# phone emb
emb = nn.Embedding(phn_len, emb_dim, padding_idx=0)
encoder_emb = nn.Embedding(grapheme_len, emb_dim, padding_idx=0)
char_lin = nn.Linear(emb_dim, d_model)
phn_lin = nn.Linear(emb_dim, d_model)
lin = nn.Linear(d_model, phn_len)
out = nn.LogSoftmax(dim = -1)

device = torch.cuda.device(0)
# model = TransformerG2P(emb,encoder_emb,char_lin,phn_lin,lin,out,d_model).cuda()

model = torch.load("g2p_model_best.pth", map_location=torch.device("cuda:0"))

char_seq = torch.LongTensor(text2grapheme(test_text)).unsqueeze(0).cuda()
char_len = torch.Tensor([char_seq.shape[1]]).cuda()
grapheme_encoded = (char_seq, char_len)



_,_,encoder_out, _ = model.forward(grapheme_encoded)
max_length = 100

def greedy_decode(model, encoder_out, end_threshold=0.6):
    # phm_idx = [get_phone_idx('_pad_')] # start
    phm_idx = []
    model.eval()
    phns_str = ""
    att_out = None
    torch.no_grad()
    for t in range(max_length):
        tgt = torch.IntTensor([get_phone_idx('<GO>')] +phm_idx ).cuda().unsqueeze(0)
        decoder_out, att_out = model.decode(tgt, encoder_out)
        logits = model.lin(decoder_out)
        p_seq = torch.softmax(logits,dim=-1)
        # print(p_seq.shape)
        if p_seq[:,-1, get_phone_idx('<END>')] > end_threshold:
            break
        phm_idx = list(torch.argmax(p_seq, dim =-1).cpu().squeeze(0).numpy())
        # print(phm_idx)
        if phm_idx[-1] == get_phone_idx('<END>'):
            break
    phns_str = "".join(["%s " % symbols[idx] for idx in phm_idx if idx !=0])

    torch.enable_grad()
    return phns_str, att_out


# 已经结束的sequence不加入到batch里
def batch_sampler(sequences, batch_size):
    batch = []
    idx = []
    for i, seq in enumerate(sequences):
        if len(seq) == 0 or seq[-1] != get_phone_idx('<END>'):
            batch.append(seq)
            idx.append(i)
        if len(batch) == batch_size:
            yield batch, idx
            batch.clear()
            idx.clear()
    yield batch, idx

def beam_search_decode(beam_size, top_k, model, encoder_out, max_batch_size = 256):
    
    go_frame = torch.LongTensor([get_phone_idx('<GO>')]).unsqueeze(0).cuda()
    model.eval()
    # torch.no_grad()
    sequences = [[]]
    probs = [1.0]
    with torch.no_grad():
        for t in range(max_length):
            tmp_sequences = []
            tmp_probs = []
            for batch, seq_idx in batch_sampler(sequences, max_batch_size):
                batch = go_frame.new_tensor(batch)
                go_frames = go_frame.repeat(batch.shape[0], 1)
                tgt = torch.cat([go_frames,batch], dim = 1)
                decoder_out, _ = model.decode(tgt, encoder_out.repeat(batch.shape[0],1,1))
                logits = model.lin(decoder_out)
                p_seq = torch.softmax(logits, dim = -1)
                # print(p_seq.shape)
                # phm_idx = torch.argmax(p_seq, dim =-1).cpu().squeeze(0).tolist()
                for j, seq in enumerate(p_seq):
                    beam_probs, indices = torch.topk(seq[-1,:], beam_size, dim = -1)
                    for b, (p, idx) in enumerate(zip(beam_probs, indices)):
                        tmp_seq = sequences[seq_idx[j]].copy() + [idx.item()]
                        tmp_prob = probs[seq_idx[j]] * p.item()
                        
                        tmp_sequences.append(tmp_seq.copy())
                        tmp_probs.append(tmp_prob)
                    
            
            probs_tensor = torch.Tensor(tmp_probs)
            _, idxs = torch.topk(probs_tensor, min(top_k, probs_tensor.shape[0]))


            sequences = [tmp_sequences[idx].copy() for idx in idxs.tolist()]
            probs = [tmp_probs[idx] for idx in idxs.tolist()]

    out_seqs = []
    for seq in sequences:
        out_seqs.append([symbols[seq[i]] for i in range(len(seq))])

    return out_seqs, probs


    # phns_str = "".join(["%s " % symbols[idx] for idx in phm_idx if idx !=0])


    # torch.enable_grad()

    # return


phn_str, att = greedy_decode(model, encoder_out, 0.5)

from matplotlib import pyplot as plt

att = att[0].tolist()
img = plt.imshow(att)
plt.imsave("att.jpg", att)
# print(beam_search_decode(2, 8, model, encoder_out))


