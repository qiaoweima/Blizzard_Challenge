import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet, get_sinusoid_encoding_table
from .modules import VarianceAdaptor, SemeticPredictor, ProsodyExtractor
from utils.tools import get_mask_from_lengths
import vector_quantize_pytorch as vq


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.enable_word_embedding = model_config['variance_predictor']['enbale_word_embed']
        self.sematic_predictor = SemeticPredictor(model_config)
        self.prosody_extractor = ProsodyExtractor(model_config, preprocess_config)

        self.encoder_size = model_config["transformer"]["encoder_hidden"]
        self.prosody_dim = model_config['prosody_extractor']['prosody_label_dim']
        self.prosody_size = model_config['prosody_extractor']['prosody_label_size']
        self.vector_quantize = vq.VectorQuantize(dim=self.prosody_dim, codebook_size=self.prosody_size)
        self.prosody_linear = nn.Linear(self.prosody_dim, self.encoder_size)
        self.finetune_word_predictor = model_config["sematic_predictor"]["finetune"]

        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def get_word_embedding(self, x, word_embeddings, word_phone_lens, mask):
        # mask, B, T1,
        # word_mask, B, T2
        #att_mask B, T1, T2
        
        embeddings = self.sematic_predictor(x, word_embeddings,word_phone_lens, mask)

        return embeddings

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        word_embedding,
        word_phone_lens,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,

    ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        if self.enable_word_embedding:
            # word_mask = get_mask_from_lengths(word_embed_len)
            prosody_prediction= self.get_word_embedding(output, word_embedding, word_phone_lens, src_masks)
            vq_prosody_predicction,_,_ = self.vector_quantize(prosody_prediction)
            out_prosody_prediction = self.prosody_linear(vq_prosody_predicction)
            
            if mels is not None:
                prosody_embeddings = self.prosody_extractor(src_masks, mels, mel_masks, d_targets, word_phone_lens)
                vq_prosody_embeddings, _ , vq_loss = self.vector_quantize(prosody_embeddings)
                out_prosody_embeddings = self.prosody_linear(vq_prosody_embeddings) 
            
            if mels is None or d_targets is None or self.finetune_word_predictor or not self.training:    
                output = output + out_prosody_prediction.detach()
            else:    
                output = output + out_prosody_embeddings

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
            word_embedding,
            word_phone_lens
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if mels is not None:
            return (
                output,
                postnet_output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens,
                prosody_prediction,
                prosody_embeddings,
                vq_loss
            )
        else:
            return (
                output,
                postnet_output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens,
            )