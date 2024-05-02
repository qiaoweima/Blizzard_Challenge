import speechbrain as sb
from speechbrain.lobes.models.g2p.model import TransformerG2P
from transformers import FlaubertModel, FlaubertTokenizer
import torch.nn as nn
from speechbrain.dataio import dataset, dataloader
from utils import symbols, graphemes, get_grapheme_idx, get_phone_idx
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
import ptvsd
from torch.utils.tensorboard import SummaryWriter

# ptvsd.enable_attach(("localhost",5600))
# ptvsd.wait_for_attach()

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
# d_model=512
# nhead=8
# num_encoder_layers=6
# num_decoder_layers=6
# d_ffn=2048
# dropout=0.1
# activation=nn.ReLU()
# custom_src_module=None
# custom_tgt_module=None
# positional_encoding='fixed_abs_sine'
# normalize_before=True
# kernel_size=15
# bias=True
# encoder_module='transformer'
# attention_type='regularMHA'
# max_length=2500
# causal=False
# pad_idx=0
# encoder_kdim=None
# encoder_vdim=None
# decoder_kdim=None
# decoder_vdim=None
# use_word_emb=False
# word_emb_enc=None

class DictDataset(dataset.Dataset):
    def __init__(self, dict_path):
        super(DictDataset,self)
        self.text_list = []
        self.phones_list = []
        with open(dict_path) as f:
            lines = f.readlines()
            for line in lines:
                text, phones = line.split("\t")
                phones = phones.strip('\n').split(' ')
                phones = [p for p in phones if p != '']
                
                
                self.text_list.append([c for c in text])
                self.phones_list.append(phones)

    def __getitem__(self, idx):
        char_idxs = [get_grapheme_idx(c) for c in self.text_list[idx]]
        phn_idxs = [get_phone_idx(p) for p in self.phones_list[idx]]
        #"Go frame"
        phn_idxs = phn_idxs
        return char_idxs, phn_idxs

    def __len__(self):
        return len(self.text_list)
    
class DictDataLoader(dataloader.DataLoader):
    def __init__(self, dict_path, batch_size, shuffule = True, drop_last=True):
        super(DictDataLoader, self).__init__(DictDataset(dict_path),batch_size, shuffle=shuffule,
                                             drop_last=drop_last,collate_fn= self.collate_fn)
        
    def collate_fn(self, batch):
        B = len(batch)
        text_seq = [torch.LongTensor(sampler[0]) for sampler in batch]
        text_lens = torch.Tensor([len(text) for text in text_seq])
        text_seq = pad_sequence(text_seq,batch_first=True)    

        phn_seq = [torch.LongTensor(sampler[1]) for sampler in batch]
        phn_lens = torch.Tensor([len(phns) + 1 for phns in phn_seq]) # 1是Go frame
        # phn_seq = pad_sequence(phn_seq, batch_first=True)

        go_frame = torch.LongTensor([get_phone_idx('<GO>')] * B).unsqueeze(1)
        phn_seq_in = pad_sequence(phn_seq, batch_first=True)
        phn_seq_in = torch.cat([go_frame, phn_seq_in], dim = 1) 
            
        end_frame = torch.LongTensor([get_phone_idx('<END>')])
        gt = [torch.cat([gt, end_frame],dim = 0) for gt in phn_seq]
        gt = pad_sequence(gt, batch_first=True)
        # gt = torch.cat([phn_seq,end_frame],dim=-1)

        return text_seq, text_lens, phn_seq_in, phn_lens, gt
    


if __name__ == '__main__':


    train_dataloader = DictDataLoader("train_dict.txt", batch_size=128)
    val_dataloader = DictDataLoader("dev_dict.txt",batch_size=256)

    val_iter = 1000
    print_iter = 100
    max_epoch = 50

    device = torch.cuda.device(0)
    model = TransformerG2P(emb,encoder_emb,char_lin,phn_lin,lin,out,d_model).cuda()
    lossfunc = nn.NLLLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = 0.001)
    sched = torch.optim.lr_scheduler.StepLR(optim, 10, 0.1)

    min_val_loss = 1000.

    writer = SummaryWriter(log_dir = "./log")
    
    def eval_acc(predict, gt):
        predict = predict.detach().cpu()
        gt = gt.detach().cpu()
        pred_y = torch.argmax(predict, dim=-1)
        mask = torch.where(pred_y == gt, 1, 0)
        acc = float(torch.sum(mask)) / torch.sum(torch.ones(gt.shape))
        return acc

    def eval_epoch(iter, model:TransformerG2P):
        
        total_loss = 0.
        total_acc = 0.
        count = 0
        model.eval()
        for batch in val_dataloader:
            text_seq, text_lens, phn_seq_in, phn_lens, gt  = batch
            grapheme_encoded = (text_seq.cuda(), text_lens.cuda())
            phn_seq_in = phn_seq_in.cuda()
            phn_lens = phn_lens.cuda()
            gt = gt.cuda()
            
            B, T = phn_seq_in.shape
            phn_encoded = (phn_seq_in.cuda(), phn_lens.cuda())
            
            # model.zero_grad()
            optim.zero_grad()

            output = model.forward(grapheme_encoded, phn_encoded)
            
            predict_y, predict_lens = output[:2]
            attention = output[-1]
            loss = lossfunc(predict_y.permute(0,2,1), gt) 
            
            total_acc += eval_acc(predict_y, gt)
            total_loss += loss
            count += 1

        total_loss /= count
        total_acc /= count    
        print("Val Iter:%d\tLoss:%f\tAcc:%f" %(iter, total_loss, total_acc))
        writer.add_scalar('val/loss', total_loss,global_step=iter)
        writer.add_scalar('val/acc', total_acc ,global_step=iter)
        # writer.add_image('alignment', attention[-1][0,:,:],global_step=iter, dataformats='HW')
        
        global min_val_loss
        if total_loss < min_val_loss:
            min_val_loss = loss
            torch.save(model, "g2p_model_best.pth")

    iter = 0
    for epoch in range(max_epoch):
        batch = None
        print("---------Epoch %d Started--------" % epoch)
        for _, batch in enumerate(train_dataloader):
            model.train()
            # print(data)
            text_seq, text_lens, phn_seq_in, phn_lens, gt  = batch
            grapheme_encoded = (text_seq.cuda(), text_lens.cuda())
            phn_seq_in = phn_seq_in.cuda()
            phn_lens = phn_lens.cuda()
            gt = gt.cuda()
            
            B, T = phn_seq_in.shape
            phn_encoded = (phn_seq_in.cuda(), phn_lens.cuda())
            
            # model.zero_grad()
            optim.zero_grad()

            output = model.forward(grapheme_encoded, phn_encoded)
            
            predict_y, predict_lens = output[:2]
            # print(phn_seq.shape, gt.shape)

            #[B, vocabsize, Seq_Long] , [B, Seq_long]
            loss = lossfunc(predict_y.permute(0,2,1), gt) #这里要跳过第一个start frame
            loss.backward()
            if iter % print_iter == 0:
                acc = eval_acc(predict_y, gt)
                print("Train Iter %d Learning Rate: %f: Loss %f , Acc %f"
                      % (iter, optim.param_groups[0]['lr'], loss.detach().cpu().numpy(), acc))
                writer.add_scalar('train/loss', loss.detach().cpu().numpy(), global_step=iter)
                writer.add_scalar('train/acc',acc,global_step=iter) 
                writer.add_scalar('train/lr', optim.param_groups[0]['lr'],global_step=iter)
            
            optim.step()
            
            iter += 1
            if iter % val_iter == 0:
                eval_epoch(iter, model)    

        sched.step()
        
        # break

        
