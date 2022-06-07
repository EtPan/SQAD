import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np

from functools import partial

if __name__ == '__main__':
    from basic import *
else:
    from .basic import *

class QARULayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QARULayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.gamma = nn.Parameter(torch.cuda.FloatTensor([1]),requires_grad=True)

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F.sigmoid()

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z        
        return h_
        
    def _attn_step(self,h_time):       
        qh_time = h_time.permute(0,3,4,1,2)
        kh_time = h_time.permute(0,3,4,2,1)
        attn = nn.Softmax(dim=3)(torch.matmul(qh_time,kh_time))
        attn = attn/ np.power(kh_time.shape[-1], 0.5)
        ah = torch.matmul(attn,qh_time).permute(0,3,4,1,2)                 
        return ah       

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []
        
        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep            
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
                ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)
        #return concatenated hidden states
        #return torch.cat(h_time, dim=2)
        
        h_time = torch.cat(h_time, dim=2)        
        #attention
        ah = self._attn_step(h_time)             
        return self.gamma*ah+h_time


class BiQLayer(QARULayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        return Z.tanh(), F1.sigmoid(), F2.sigmoid()

    def forward(self, inputs, fname=None):        
        h = None 
        Z, F1, F2 = self._conv_step(inputs)
        hsl = [] ; hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep            
            h = self._rnn_step(z, f, h)
            hsl.append(h)
        
        h = None
        for time, (z, f) in enumerate((zip(reversed(zs), reversed(F2.split(1, 2))))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)
        
        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {'z':Z, 'fl':F1, 'fr':F2, 'hsl':hsl, 'hsr':hsr}
            torch.save(stats_dict, fname)
        return hsl + hsr


class BiQ(BiQLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(BiQ, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*3, k, s, p, bn=bn), act=act)


class BiDeQ(BiQLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bias=False, bn=True, act='tanh'):
        super(BiDeQ, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels*3, k, s, p, bias=bias, bn=bn), act=act)


class QARU(QARULayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QARU, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn), act=act)


class DeQ(QARULayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(DeQ, self).__init__(
            in_channels, hidden_channels, BasicDeConv3d(in_channels, hidden_channels*2, k, s, p, bn=bn), act=act)


class UpQ(QARULayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1,2,2), bn=True, act='tanh'):
        super(UpQ, self).__init__(
            in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels*2, k, s, p, upsample, bn=bn), act=act)


class QAE(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, 
    BiQ=None, BiDeQ=None,QEncoder=None, QDecoder=None, bn=True, act='tanh'):
        super(QAE, self).__init__()
        assert sample_idx is None or isinstance(sample_idx, list)

        if sample_idx is None: sample_idx = []
        self.feature_extractor = BiQ(in_channels, channels, bn=bn, act=act)
        self.encoder = QEncoder(channels, num_half_layer, sample_idx, bn=bn, act=act)
        self.decoder = QDecoder(channels*(2**len(sample_idx)), num_half_layer, sample_idx, bn=bn, act=act)
        self.reconstructor = BiDeQ(channels, in_channels, bias=True, bn=bn, act=act)
        
    def forward(self, x):
        xs = [x]
        out = self.feature_extractor(x)
        xs.append(out)                 
        out, reverse = self.encoder(out, xs, reverse=False)
        out = self.decoder(out, xs, reverse=(reverse))               
        out = self.reconstructor(out)
        out = out + xs.pop()
        return out


class QEncoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, QARU=None, 
    bn=True, act='tanh'):
        super(QEncoder, self).__init__()
        # Encoder        
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = QARU(channels, channels, bn=bn, act=act)
            else:
                encoder_layer = QARU(channels, 2*channels, k=3, s=(1,2,2), p=1, bn=bn, act=act)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse=False):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer-1):
            x = self.layers[i](x, reverse=reverse)
            reverse = not reverse
            xs.append(x)  
        x = self.layers[-1](x, reverse=reverse)
        reverse = not reverse
        return x, reverse


class QDecoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, DeQ=None, UpQ=None, 
    bn=True, act='tanh'):
        super(QDecoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = DeQ(channels, channels, bn=bn, act=act)
            else:
                decoder_layer = UpQ(channels, channels//2, bn=bn, act=act)  
                channels //= 2
            cat_layer = CatBlock(channels*2,channels)
            self.layers.append(decoder_layer)
            self.layers.append(cat_layer)

    def forward(self, x, xs, reverse=False):        
        num_half_layer = len(self.layers)
        for i in range(0, num_half_layer,2):
            x = self.layers[i](x, reverse=reverse)
            x = torch.cat((x,xs.pop()),dim=1)
            x = self.layers[i+1](x)
        return x

Encoder = partial(QEncoder, QARU=QARU)

Decoder = partial(QDecoder, DeQ=DeQ, UpQ=UpQ)

SQAD = partial(QAE, 
               BiQ=BiQ, BiDeQ=BiDeQ,
               QEncoder=Encoder,QDecoder=Decoder)

