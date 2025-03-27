#coding:utf-8

import os
import os.path as osp

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

# from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution
# from Modules.diffusion.modules import Transformer1d, StyleTransformer1d
# from Modules.diffusion.diffusion import AudioDiffusionConditional

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from munch import Munch
import yaml

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)

class LearnedUpSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
    
        return s

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x) 
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p
        
        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if self.downsample_type == 'none':
            return x
        else:
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
            
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
    
class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        # self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask



class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))
        
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class ProsodyPredictor(nn.Module):

    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__() 
        
        self.text_encoder = DurationEncoder(sty_dim=style_dim, 
                                            d_model=d_hid,
                                            nlayers=nlayers, 
                                            dropout=dropout)

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList()
        self.F0.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.F0.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))

        self.N = nn.ModuleList()
        self.N.append(AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout))
        self.N.append(AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout))
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)


    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        batch_size = d.shape[0]
        text_size = d.shape[1]
        
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        
        m = m.to(text_lengths.device).unsqueeze(1)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])

        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
                
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        
        en = (d.transpose(-1, -2) @ alignment)

        return duration.squeeze(-1), en
    
    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        
        return F0.squeeze(1), N.squeeze(1)
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
class DurationEncoder(nn.Module):

    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, 
                                 d_model // 2, 
                                 num_layers=1, 
                                 batch_first=True, 
                                 bidirectional=True, 
                                 dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        
        
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
                
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad.to(x.device)
        
        return x.transpose(-1, -2)
    
    def inference(self, x, style):
        x = self.embedding(x.transpose(-1, -2)) * math.sqrt(self.d_model)
        style = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, style], axis=-1)
        src = self.pos_encoder(x)
        output = self.transformer_encoder(src).transpose(0, 1)
        return output
    
    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG, adapt_vocab=True):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path, adapt_vocab=True):
        # Load saved parameters first
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        saved_params = checkpoint['model']
        
        # Extract original vocabulary size from checkpoint
        original_n_symbols = None
        for key in saved_params:
            if 'embedding.weight' in key:
                original_n_symbols = saved_params[key].size(0)
                break
        
        # Either adapt the model config or proceed with adaptation later
        if not adapt_vocab and original_n_symbols is not None:
            # Force the model to use the same vocabulary size as the checkpoint
            if 'n_symbols' in model_config:
                print(f"Setting vocabulary size in model config to {original_n_symbols}")
                model_config['n_symbols'] = original_n_symbols
        
        # Create model with configuration
        model = ASRCNN(**model_config)
        
        # Get current vocabulary size from model
        current_n_symbols = None
        for key, param in model.state_dict().items():
            if 'embedding.weight' in key:
                current_n_symbols = param.size(0)
                break
        
        if adapt_vocab and original_n_symbols != current_n_symbols:
            print(f"Model vocabulary size: {current_n_symbols}")
            print(f"Checkpoint vocabulary size: {original_n_symbols}")
            
            # Directly modify problematic parameters in the state dict
            for key in list(saved_params.keys()):
                # Skip parameters that aren't vocabulary-dependent
                if not any(pattern in key for pattern in [
                    'embedding.weight',
                    'project_to_n_symbols',
                    'ctc_linear.2.linear_layer'
                ]):
                    continue
                
                param = saved_params[key]
                if param.dim() == 2:
                    # Handle 2D parameters (weight matrices)
                    if key.endswith('.weight'):
                        if param.size(0) == original_n_symbols:
                            # Output dimension (like projection layers)
                            print(f"Adapting output dimension of {key} from {param.size(0)} to {current_n_symbols}")
                            new_param = torch.zeros((current_n_symbols, param.size(1)), 
                                                   device=param.device, dtype=param.dtype)
                            # Copy existing weights
                            min_size = min(original_n_symbols, current_n_symbols)
                            new_param[:min_size] = param[:min_size]
                            # Initialize new weights if vocabulary expanded
                            if current_n_symbols > original_n_symbols:
                                # Use normal initialization for the new rows
                                torch.nn.init.normal_(new_param[original_n_symbols:], mean=0, std=0.02)
                            saved_params[key] = new_param
                        
                        elif param.size(1) == original_n_symbols:
                            # Input dimension
                            print(f"Adapting input dimension of {key} from {param.size(1)} to {current_n_symbols}")
                            new_param = torch.zeros((param.size(0), current_n_symbols), 
                                                   device=param.device, dtype=param.dtype)
                            # Copy existing weights
                            min_size = min(original_n_symbols, current_n_symbols)
                            new_param[:, :min_size] = param[:, :min_size]
                            # Initialize new weights if vocabulary expanded
                            if current_n_symbols > original_n_symbols:
                                # Use normal initialization for the new columns
                                torch.nn.init.normal_(new_param[:, original_n_symbols:], mean=0, std=0.02)
                            saved_params[key] = new_param
                
                elif param.dim() == 1:
                    # Handle 1D parameters (bias vectors)
                    if param.size(0) == original_n_symbols:
                        print(f"Adapting bias vector {key} from size {param.size(0)} to {current_n_symbols}")
                        new_param = torch.zeros(current_n_symbols, device=param.device, dtype=param.dtype)
                        # Copy existing biases
                        min_size = min(original_n_symbols, current_n_symbols)
                        new_param[:min_size] = param[:min_size]
                        # Initialize new biases if vocabulary expanded
                        if current_n_symbols > original_n_symbols:
                            new_param[original_n_symbols:].fill_(0)  # Initialize new biases with zeros
                        saved_params[key] = new_param
        
        # Try to load the state dict
        try:
            # First try with filtered dict - only include parameters that match in size
            model_dict = model.state_dict()
            filtered_params = {k: v for k, v in saved_params.items() 
                              if k in model_dict and v.size() == model_dict[k].size()}
            
            # Check if we've missed any important parameters
            missing_params = [k for k in model_dict.keys() 
                             if k in saved_params and k not in filtered_params]
            if missing_params:
                print(f"Warning: The following parameters couldn't be loaded due to size mismatch:")
                for param in missing_params:
                    if param in saved_params:
                        print(f"  {param}: model shape {model_dict[param].size()} vs checkpoint shape {saved_params[param].size()}")
            
            # Load the filtered parameters
            model.load_state_dict(filtered_params, strict=False)
            print("ASR model loaded with compatible parameters only")
            print("-------------------------------")
            
        except Exception as e:
            print(f"Error during final model loading: {e}")
            print("Model initialization failed. Please check your model configuration.")
            return None
        
        return model

    # Load config and model
    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH, adapt_vocab)
    if asr_model is not None:
        _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, pitch_extractor, bert):
    assert args.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'
    
    if args.decoder.type == "istftnet":
        from Modules.istftnet import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes, 
                gen_istft_n_fft=args.decoder.gen_istft_n_fft, gen_istft_hop_size=args.decoder.gen_istft_hop_size) 
    else:
        from Modules.hifigan import Decoder
        decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                resblock_kernel_sizes = args.decoder.resblock_kernel_sizes,
                upsample_rates = args.decoder.upsample_rates,
                upsample_initial_channel=args.decoder.upsample_initial_channel,
                resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                upsample_kernel_sizes=args.decoder.upsample_kernel_sizes) 
        
    text_encoder = TextEncoder(channels=args.hidden_dim, kernel_size=5, depth=args.n_layer, n_symbols=args.n_token)
    
    predictor = ProsodyPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout)
    
    style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # acoustic style encoder
    predictor_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim) # prosodic style encoder
        
    # define diffusion model
    # if args.multispeaker:
    #     transformer = StyleTransformer1d(channels=args.style_dim*2, 
    #                                 context_embedding_features=bert.config.hidden_size,
    #                                 context_features=args.style_dim*2, 
    #                                 **args.diffusion.transformer)
    # else:
    #     transformer = Transformer1d(channels=args.style_dim*2, 
    #                                 context_embedding_features=bert.config.hidden_size,
    #                                 **args.diffusion.transformer)
    
    # diffusion = AudioDiffusionConditional(
    #     in_channels=1,
    #     embedding_max_length=bert.config.max_position_embeddings,
    #     embedding_features=bert.config.hidden_size,
    #     embedding_mask_proba=args.diffusion.embedding_mask_proba, # Conditional dropout of batch elements,
    #     channels=args.style_dim*2,
    #     context_features=args.style_dim*2,
    # )
    
    # diffusion.diffusion = KDiffusion(
    #     net=diffusion.unet,
    #     sigma_distribution=LogNormalDistribution(mean = args.diffusion.dist.mean, std = args.diffusion.dist.std),
    #     sigma_data=args.diffusion.dist.sigma_data, # a placeholder, will be changed dynamically when start training diffusion model
    #     dynamic_threshold=0.0 
    # )
    # diffusion.diffusion.net = transformer
    # diffusion.unet = transformer

    
    nets = Munch(
            bert=bert,
            bert_encoder=nn.Linear(bert.config.hidden_size, args.hidden_dim),

            predictor=predictor,
            decoder=decoder,
            text_encoder=text_encoder,

            predictor_encoder=predictor_encoder,
            style_encoder=style_encoder,
            # diffusion=diffusion,

            text_aligner = text_aligner,
            pitch_extractor=pitch_extractor,

            mpd = MultiPeriodDiscriminator(),
            msd = MultiResSpecDiscriminator(),
        
            # # slm discriminator head
            wd = WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel),
       )
    
    return nets

def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    # params = state['net']
    params = state

    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # print(k)
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    print('params weight:')
    print(params['decoder']['decode.0.conv1.weight_g'][0])
    print('model weight:')
    print(model.decoder.decode[0].conv1.weight_g[0], model.decoder.decode[0].conv1.weight_g[0].device)
    print('params bias:')
    print(params['decoder']['decode.0.conv1.bias'][0])
    print('model bias:')
    print(model.decoder.decode[0].conv1.bias[0], model.decoder.decode[0].conv1.bias[0].device)

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        poch_iters = state["poch_iters"] if "poch_iters" in state else 0
    else:
        epoch = 0
        iters = 0
        poch_iters = 0

    return model, optimizer, epoch, iters, poch_iters

def load_checkpoint_hf(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="SirAB/kokoro_finetune_v1", filename=path)

    state = torch.load(model_path, map_location='cpu')
    params = state['net']

    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    # print(k)
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    print('params weight:')
    print(params['decoder']['decode.0.conv1.weight_g'][0])
    print('model weight:')
    print(model.decoder.decode[0].conv1.weight_g[0], model.decoder.decode[0].conv1.weight_g[0].device)
    print('params bias:')
    print(params['decoder']['decode.0.conv1.bias'][0])
    print('model bias:')
    print(model.decoder.decode[0].conv1.bias[0], model.decoder.decode[0].conv1.bias[0].device)

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        poch_iters = state["poch_iters"] if "poch_iters" in state else 0
    else:
        epoch = 0
        iters = 0
        poch_iters = state["poch_iters"] if "poch_iters" in state else 0

    return model, optimizer, epoch, iters, poch_iters

def load_checkpoint_kokoro(model, optimizer, path2, load_only_params=False, ignore_modules=[], adapt_embedding=True):
    # Load first model state (kokoro)
    from huggingface_hub import hf_hub_download
    kokoro_model = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth")
    state1 = torch.load(kokoro_model, map_location='cpu')
    params1 = state1

    # Load second model state (styletts2 checkpoint)
    styletts_model = hf_hub_download(repo_id="yl4579/StyleTTS2-LibriTTS", filename="Models/LibriTTS/epochs_2nd_00020.pth")
    state2 = torch.load(styletts_model, map_location='cpu')
    params2 = state2['net']
    
    # Track which modules were loaded from the first model
    loaded_modules = []
    
    # Get dimensions information for adaptation
    original_n_symbols = 178  # From error message
    current_n_symbols = 185   # From error message
    
    # Function to adapt embedding dimensions
    def adapt_tensor_dimensions(tensor, is_embedding=True):
        if tensor.dim() == 2 and tensor.size(0) == original_n_symbols:
            # This is likely an embedding or projection weight
            embed_dim = tensor.size(1)
            new_tensor = torch.zeros(current_n_symbols, embed_dim, device=tensor.device)
            
            # Copy original weights
            new_tensor[:original_n_symbols] = tensor
            
            # Initialize new embeddings (for the added symbols)
            with torch.no_grad():
                mean = tensor.mean()
                std = tensor.std()
                torch.nn.init.normal_(new_tensor[original_n_symbols:], mean=mean, std=std)
                
            return new_tensor
        elif tensor.dim() == 1 and tensor.size(0) == original_n_symbols:
            # This is likely a bias vector
            new_tensor = torch.zeros(current_n_symbols, device=tensor.device)
            
            # Copy original weights
            new_tensor[:original_n_symbols] = tensor
            
            # Initialize new biases
            with torch.no_grad():
                mean = tensor.mean()
                std = tensor.std()
                torch.nn.init.normal_(new_tensor[original_n_symbols:], mean=mean, std=std)
                
            return new_tensor
        else:
            # No dimension adaptation needed
            return tensor

    # Function to process state dict for a module
    def process_state_dict(state_dict, adapt_dims=True):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            name = k[7:] if k.startswith('module.') else k
            
            if adapt_dims and any(emb_key in name for emb_key in [
                'embedding.weight', 
                'word_embeddings.weight',
                'linear_layer.weight',
                'linear_layer.bias',
                'project_to_n_symbols.weight',
                'project_to_n_symbols.bias'
            ]):
                new_state_dict[name] = adapt_tensor_dimensions(v)
            else:
                new_state_dict[name] = v
                
        return new_state_dict

    # First load from model 1
    for key in model:
        if key in params1 and key not in ignore_modules:
            try:
                # Special handling for modules that need dimension adaptation
                if key in ['text_encoder', 'bert'] and adapt_embedding:
                    state_dict = process_state_dict(params1[key], adapt_dims=True)
                    model[key].load_state_dict(state_dict, strict=False)
                    loaded_modules.append(key)
                    print(f'{key} loaded from first model with dimension adaptation')
                else:
                    model[key].load_state_dict(params1[key])
                    loaded_modules.append(key)
                    print(f'{key} loaded from first model')
            except Exception as e:
                try:
                    state_dict = process_state_dict(params1[key], adapt_dims=(key in ['text_encoder', 'bert'] and adapt_embedding))
                    model[key].load_state_dict(state_dict, strict=False)
                    loaded_modules.append(key)
                    print(f'{key} loaded from first model with strict=False')
                except Exception as e2:
                    print(f"Failed to load {key} from first model: {e2}")

    # Then load missing modules from model 2
    for key in model:
        if key in params2 and key not in ignore_modules and key not in loaded_modules:
            try:
                # Special handling for text_aligner which needs dimension adaptation
                if key == 'text_aligner' and adapt_embedding:
                    state_dict = process_state_dict(params2[key], adapt_dims=True)
                    model[key].load_state_dict(state_dict, strict=False)
                    loaded_modules.append(key)
                    print(f'{key} loaded from second model with dimension adaptation')
                else:
                    model[key].load_state_dict(params2[key])
                    loaded_modules.append(key)
                    print(f'{key} loaded from second model')
            except Exception as e:
                try:
                    state_dict = process_state_dict(params2[key], adapt_dims=(key == 'text_aligner' and adapt_embedding))
                    model[key].load_state_dict(state_dict, strict=False)
                    loaded_modules.append(key)
                    print(f'{key} loaded from second model with strict=False')
                except Exception as e2:
                    print(f"Failed to load {key} from second model: {e2}")

    print("---------------------------")

    # Validation checks
    if 'decoder' in model and 'decoder' in params1:
        try:
            print('DECODER VALIDATION:')
            print('params weight:')
            print(params1['decoder']['module.decode.0.conv1.weight_g'][0])
            print('model weight:')
            print(model['decoder'].decode[0].conv1.weight_g[0], model['decoder'].decode[0].conv1.weight_g[0].device)
            print('params bias:')
            print(params1['decoder']['module.decode.0.conv1.bias'][0])
            print('model bias:')
            print(model['decoder'].decode[0].conv1.bias[0], model['decoder'].decode[0].conv1.bias[0].device)
        except Exception as e:
            print(f"Could not print decoder validation weights: {e}")
    print("---------------------------")

    # Validation for bert module
    if 'bert' in model and 'bert' in params1:
        try:
            print('\nBERT VALIDATION:')
            # Check for word embeddings - validate dimension adaptation worked
            original_embed = None
            if 'module.embeddings.word_embeddings.weight' in params1['bert']:
                original_embed = params1['bert']['module.embeddings.word_embeddings.weight']
            elif 'embeddings.word_embeddings.weight' in params1['bert']:
                original_embed = params1['bert']['embeddings.word_embeddings.weight']
            
            if original_embed is not None:
                print('Original embedding shape:', original_embed.shape)
                print('Original embedding second row:', original_embed[1][:5])  # First 5 values
                print('Original embedding last row:', original_embed[-1][:5])  # First 5 values
                
                # Check model's current embedding
                current_embed = model['bert'].embeddings.word_embeddings.weight
                print('Current embedding shape:', current_embed.shape)
                print('Current embedding second row:', current_embed[1][:5])  # First 5 values
                print('Current embedding last row before change:', current_embed[-8][:5])  # First 5 values
                
                # Check newly added embeddings
                if current_embed.shape[0] > original_embed.shape[0]:
                    print(f'Added embeddings (checking row {original_embed.shape[0]}):', 
                          current_embed[original_embed.shape[0]-1][:5])
        except Exception as e:
            print(f"Could not print bert validation weights: {e}")
    print("---------------------------")

    # Validation for text_aligner module
    if 'text_aligner' in model and 'text_aligner' in params2:
        try:
            print('\nTEXT_ALIGNER VALIDATION:')
            # Check for CTC linear layer
            original_weight = None
            if 'module.ctc_linear.2.linear_layer.weight' in params2['text_aligner']:
                original_weight = params2['text_aligner']['module.ctc_linear.2.linear_layer.weight']
            elif 'ctc_linear.2.linear_layer.weight' in params2['text_aligner']:
                original_weight = params2['text_aligner']['ctc_linear.2.linear_layer.weight']
            
            if original_weight is not None:
                print('Original CTC linear weight shape:', original_weight.shape)
                print('Original CTC first row:', original_weight[0][:5])  # First 5 values
                print('Original CTC last row:', original_weight[-1][:5])  # First 5 values
                
                # Check model's current CTC weight
                current_weight = model['text_aligner'].ctc_linear[2].linear_layer.weight
                print('Current CTC weight shape:', current_weight.shape)
                print('Current CTC first row:', current_weight[0][:5])  # First 5 values
                print('Current CTC last row:', current_weight[-8][:5])  # First 5 values
                
                # Check embedding layer too
                if hasattr(model['text_aligner'], 'asr_s2s') and hasattr(model['text_aligner'].asr_s2s, 'embedding'):
                    print('\nText aligner embedding validation:')
                    current_embed = model['text_aligner'].asr_s2s.embedding.weight
                    print('Current embedding shape:', current_embed.shape)
                    
                    # Also check project_to_n_symbols
                    if hasattr(model['text_aligner'].asr_s2s, 'project_to_n_symbols'):
                        print('\nText aligner projection validation:')
                        current_proj = model['text_aligner'].asr_s2s.project_to_n_symbols.weight
                        print('Current projection shape:', current_proj.shape)
        except Exception as e:
            print(f"Could not print text_aligner validation weights: {e}")
    print("---------------------------")

    # Validation for text_encoder module 
    if 'text_encoder' in model and 'text_encoder' in params1:
        try:
            print('\nTEXT_ENCODER VALIDATION:')
            original_embed = None
            if 'module.embedding.weight' in params1['text_encoder']:
                original_embed = params1['text_encoder']['module.embedding.weight']
            elif 'embedding.weight' in params1['text_encoder']:
                original_embed = params1['text_encoder']['embedding.weight']
            
            if original_embed is not None:
                print('Original embedding shape:', original_embed.shape)
                print('Original embedding first row:', original_embed[0][:5])  # First 5 values
                print('Original embedding last row:', original_embed[-1][:5])  # First 5 values

                # Check model's current embedding
                current_embed = model['text_encoder'].embedding.weight
                print('Current embedding shape:', current_embed.shape)
                print('Current embedding first row:', current_embed[0][:5])  # First 5 values
                print('Current embedding last row before:', current_embed[-8][:5])  # First 5 values

                
                # Check newly added embeddings
                if current_embed.shape[0] > original_embed.shape[0]:
                    print(f'Added embeddings (checking row {original_embed.shape[0]}):', 
                          current_embed[original_embed.shape[0]-1][:5])
        except Exception as e:
            print(f"Could not print text_encoder validation weights: {e}")
    print("---------------------------")

    # Set all modules to eval mode
    _ = [model[key].eval() for key in model]
    
    # Handle optimizer and training state
    # if not load_only_params:
    #     # Prioritize training state from first model
    #     epoch = state2.get("epoch", 0)
    #     iters = state2.get("iters", 0)
    #     if "optimizer" in state1:
    #         optimizer.load_state_dict(state1["optimizer"])
    #     elif "optimizer" in state2:
    #         optimizer.load_state_dict(state2["optimizer"])
    #     poch_iters = state2.get("poch_iters", 0)
    # else:
    epoch = 0
    iters = 0
    poch_iters = 0
    
    # Report which modules weren't loaded at all
    all_modules = set(model.keys()) - set(ignore_modules)
    missing_modules = all_modules - set(loaded_modules)
    if missing_modules:
        print(f"Warning: The following modules were not loaded from either model: {missing_modules}")
    
    return model, optimizer, epoch, iters, poch_iters