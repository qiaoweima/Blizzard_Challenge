import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from torch.nn.utils.rnn import pad_sequence
from transformer.SubLayers import MultiHeadAttention
from transformer import get_sinusoid_encoding_table


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        # self.semetic_predictor = SemeticPredictor(model_config)
        # self.prosody_extractor = ProsodyExtractor(model_config, preprocess_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    


    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        word_embedding = None,
        word_embed_len = None
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


#semantic predictor
class SemeticPredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(SemeticPredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        # self.dropout = model_config["variance_predictor"]["dropout"]
        self.dropout = 0.2
        
        self.prosody_dim = model_config['prosody_extractor']['prosody_label_dim']
        # self.prosody_size = model_config['prosody_extractor']['prosody_label_size']
        
        self.embedding_size = model_config["variance_predictor"]["bert_embedding_size"]

        self.input_linear = nn.Linear(self.embedding_size, self.input_size)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(512, self.input_size).unsqueeze(0),
            requires_grad=False,
        )       

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2, #unchange T
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,  #unchange T
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, self.prosody_dim)
    
    def expand_word_embedding(self, embeddngs, sentence_word_phone_len, pad_len):
        embeds =[]
        for i, phone_len in enumerate(sentence_word_phone_len):
            if phone_len != 0:
                embeds.append(embeddngs[i:i+1,:].expand(phone_len,-1))

        embeds = torch.cat(embeds, dim =0)
        assert(embeds.shape[0] == sum(sentence_word_phone_len))
        if embeds.shape[0] < pad_len:
            embeds = torch.cat([embeds, 
                                embeds.new_zeros((pad_len - embeds.shape[0], embeds.shape[1]))], 
                            dim = 0)

        return embeds

    #word level prosody predictor,
    def forward(self, encoder_output, sentence_embeddings, word_phone_lens, mask):
        # encoder_output = encoder_output.detach()
        B,T,C = sentence_embeddings.shape
        # if encoder_output.shape[1] != sentence_embeddings.shape[1]:
            # print(encoder_output.shape, sentence_embeddings.shape)

        sentence_embeddings = self.input_linear(sentence_embeddings )
        # print(encoder_output.shape,att_in.shape)

        out = self.conv_layer(sentence_embeddings)
        out = self.linear_layer(out)
        # out = out.squeeze(-1)

        expand_out = []
        for b, batch in enumerate(out):
            expand_out.append(self.expand_word_embedding(batch, word_phone_lens[b], pad_len = encoder_output.shape[1]))
        
        out = torch.stack(expand_out, dim =0)

        mask = mask.unsqueeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class ProsodyExtractor(nn.Module):

    def __init__(self, model_config, preprocess_config):
        super(ProsodyExtractor, self).__init__()

        self.input_size = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.filter_size = model_config["prosody_extractor"]["filter_size"]
        self.kernel = model_config["prosody_extractor"]["kernel_size"]
        self.conv_output_size = model_config["prosody_extractor"]["filter_size"]
        self.dropout = model_config["prosody_extractor"]["dropout"]
        self.mel_channel = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.prosody_dim = model_config['prosody_extractor']['prosody_label_dim']

        # self.input_linear = nn.Linear(self.embedding_size, self.input_size)

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel[0][0],
                            padding=(self.kernel[0][0] - 1) // 2,
                            # padding=[(self.kernel[0][0] - 1) // 2,
                            #          (self.kernel[0][1] - 1) // 2], #unchange T
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel[1][0],
                            padding=(self.kernel[0][0] - 1) // 2,
                            # padding=[(self.kernel[1][0] - 1) // 2,
                            #          (self.kernel[1][1] - 1) // 2],  #unchange T
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                    # ("flatten", Flatten()),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.filter_size, self.conv_output_size)
        self.drop_out = nn.Dropout(self.dropout)
        self.linear_layer_2 = nn.Linear(self.conv_output_size, self.prosody_dim)

    def token_wise_mean_pool(self, mels, durations):
        batches = []
        for batch, duration in zip(mels, durations):
            out = []
            start = 0
            for d in duration:
                if d >0:
                    out.append(torch.mean(batch[start:start+d,:],dim=0))
                elif pad:
                    out.append(batch.new_zeros([batch.shape[1]]))
                start += d
            out = torch.stack(out, dim = 0)
            batches.append(out)
        batches = torch.stack(batches, dim=0)
        
        return batches

    def mean_pool_word_level(self, phone_embed, word_phone_len):
        batches = []
        for batch, phone_len in zip(phone_embed, word_phone_len):
            out = []
            start = 0
            for l in phone_len:
                end = start + l
                if l > 0:
                    out.append(torch.mean(batch[start:end,:], dim =0))
                start = end
            out = torch.stack(out, dim = 0)
            batches.append(out)
        batches = pad_sequence(batches, batch_first=True)        
        return batches

    def expand_word_embedding(self, embeddngs, sentence_word_phone_len):
        embeds =[]
        start = 0
        # print(embeddngs.shape, sum(sentence_word_phone_len))
        for i, phone_len in enumerate(sentence_word_phone_len):
            if phone_len != 0:
                embeds.append(embeddngs[start:start+1,:].expand(phone_len,-1))
                start += 1
        embeds = torch.cat(embeds, dim =0)
        assert(embeds.shape[0] == sum(sentence_word_phone_len))

        return embeds

    def forward(self, mask, mels, mel_mask, durations, word_phone_len):
        # mels = mels.unsqueeze(-1)
        x = self.conv_layer(mels)
        x = self.token_wise_mean_pool(x, durations) # phone level
        x = self.mean_pool_word_level(x, word_phone_len)
        x = self.linear_layer(x)
        x = torch.relu(x)
        x = self.drop_out(x)
        x = self.linear_layer_2(x)
        x = torch.relu(x)
        x = self.drop_out(x)

        expand_x = []
        for b in range(x.shape[0]):
            expand_x.append(self.expand_word_embedding(x[b,:,:], word_phone_len[b]))
        
        x = pad_sequence(expand_x, batch_first=True)   

        
        mask = mask.unsqueeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        
        return x

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x



class Conv2D(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1,1),
        stride=(1,1),
        padding=(0,0),
        dilation=(1,1),
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv2D, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().permute(0,3,1,2)
        x = self.conv(x)
        x = x.contiguous().permute(0,2,3,1)

        return x
    



class Flatten(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Flatten, self).__init__()


    def forward(self, x):
        B, T, Bins, C = x.shape 
        x = x.contiguous().reshape(B, T, -1)

        return x
    


