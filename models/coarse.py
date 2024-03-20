
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat , batch_transform
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.emb_dims = 1216
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = 1216
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, f_x , f_y):
        src = f_x
        tgt = f_y
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding








class PointNet(nn.Module):
    def __init__(self, in_dim, gn, out_dims):
        super(PointNet, self).__init__()
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0)) 
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                         nn.GroupNorm(8, out_dim))
            self.backbone.add_module(f'pointnet_relu_{i}',
                                    nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone(x)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)
        return f, g
    
class PointNet1(nn.Module):
    def __init__(self, in_dim,gn):
        super(PointNet1 , self).__init__()
        if gn:
            self.conv_block1 = nn.Sequential(nn.Conv1d(in_dim, 64, 1,1,0),  nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
            self.conv_block2 = nn.Sequential(nn.Conv1d(64, 64, 1,1,0),  nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
            self.conv_block3 = nn.Sequential(nn.Conv1d(64, 64, 1,1,0),  nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
            self.conv_block4 = nn.Sequential(nn.Conv1d(64, 256, 1,1,0),  nn.GroupNorm(8, 256), nn.ReLU(inplace=True))
            self.conv_block5 = nn.Sequential(nn.Conv1d(256, 1024, 1,1,0),  nn.GroupNorm(8, 1024), nn.ReLU(inplace=True))
        else:
            self.conv_block1 = nn.Sequential(nn.Conv1d(in_dim, 64, 1,1,0),  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
            self.conv_block2 = nn.Sequential(nn.Conv1d(64, 64, 1,1,0),  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
            self.conv_block3 = nn.Sequential(nn.Conv1d(64, 64, 1,1,0),  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
            self.conv_block4 = nn.Sequential(nn.Conv1d(64, 256, 1,1,0),  nn.BatchNorm1d(256), nn.ReLU(inplace=True))
            self.conv_block5 = nn.Sequential(nn.Conv1d(256, 1024, 1,1,0),  nn.BatchNorm1d(1024), nn.ReLU(inplace=True))
    def forward(self, x, pooling=True):
        point1_feat64 = self.conv_block1(x)
        point2_feat64 = self.conv_block2(point1_feat64)
        point3_feat64 = self.conv_block3(point2_feat64)
        x = self.conv_block4(point3_feat64)
        point_feat1024 = self.conv_block5(x)
        L = torch.cat([point1_feat64,point2_feat64,point3_feat64,point_feat1024], dim = 1)
        if not pooling:
            return L
        x, _ = torch.max(L, dim=2)
        return L , x


   
class PointNet_de(nn.Module):
    def __init__(self, in_dim, gn, out_dims):
        super(PointNet_de, self).__init__()
        self.backbone_de = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone_de.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if out_dim != 2:
                if gn:
                    self.backbone_de.add_module(f'gn_{i}',nn.GroupNorm(8, out_dim))
                self.backbone_de.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone_de(x)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)
        return f, g






class MLPs(nn.Module):
    def __init__(self, in_dim , gn , mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()
        for i, out_dim in enumerate(mlps):
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if out_dim != 7 :
                if gn:
                    self.mlps.add_module(f'gn_{i}',nn.GroupNorm(8, out_dim))
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)
        return x
    






class InitModule(nn.Module):
    def __init__(self, in_dim, gn):
        super(InitModule, self).__init__()
#        self.encoder = PointNet(in_dim=in_dim,
#                                gn=gn,
#                                out_dims=[64, 64, 64, 128, 1024])
#        out_dims=[64, 64, 64, 128, 1024]
        self.encoder1 = PointNet1(in_dim=in_dim,gn=gn)
        
        self.decoder_ol = PointNet_de(in_dim=6080,
                                   gn=gn,
                                   out_dims=[2048,2048,512,256,2])
#       [2048, 2048,512, 256, 2]
        self.decoder_qt = MLPs(in_dim=2432,
                               gn = gn,
                               mlps=[1024,1024,256,256,7])
        self.pointer = Transformer()
#        [2048,2048,256,256,7]

    def forward(self, src,src1, src2, tgt):
        '''
        Context-Guided Model for initial alignment and overlap score.
        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :return: T0: (B, 3, 4), OX: (B, N, 2), OY: (B, M, 2)
        '''
        x = src.permute(0, 2, 1).contiguous()
        x1 = src1.permute(0, 2, 1).contiguous()
        x2 = src2.permute(0, 2, 1).contiguous()
        y = tgt.permute(0, 2, 1).contiguous()
        
        f_x, g_x = self.encoder1(x)
        f_x2, g_x2 = self.encoder1(x1)
        f_x3, g_x3 = self.encoder1(x2)
        f_y, g_y = self.encoder1(y)

        concat = torch.cat((g_x, g_y), dim=1)
        # regression initial alignment
        out = self.decoder_qt(concat)
        batch_t, batch_quat = out[:, :3], out[:, 3:] / (
                torch.norm(out[:, 3:], dim=1, keepdim=True) + 1e-8)
        batch_R = batch_quat2mat(batch_quat)
        batch_T = torch.cat([batch_R, batch_t[..., None]], dim=-1)
        transformed_x = batch_transform(x.permute(0, 2, 1).contiguous(),batch_R, batch_t)
        # overlap prediction
        f_x1, g_x1 = self.encoder1(transformed_x.permute(0, 2, 1).contiguous())
        ff_x , ff_y = self.pointer(f_x1,f_y)                   #1024维度
        g_x_expand = torch.unsqueeze(g_x1, dim=-1).expand_as(f_x1)
        g_y_expand = torch.unsqueeze(g_y, dim=-1).expand_as(f_y)
        f_x_ensemble = torch.cat([f_x1, g_x_expand, g_y_expand, g_x_expand - g_y_expand,ff_x], dim=1)
        f_y_ensemble = torch.cat([f_y, g_y_expand, g_x_expand, g_y_expand - g_x_expand, ff_y], dim=1)

        x_ol = self.decoder_ol(f_x_ensemble, pooling=False)
        y_ol = self.decoder_ol(f_y_ensemble, pooling=False)
        
        concat1 = torch.cat((g_x2, g_y), dim=1)
        # regression initial alignment
        out1 = self.decoder_qt(concat1)
        batch_t1, batch_quat1 = out1[:, :3], out1[:, 3:] / (
                torch.norm(out1[:, 3:], dim=1, keepdim=True) + 1e-8)
        batch_R1 = batch_quat2mat(batch_quat1)
        batch_T1 = torch.cat([batch_R1, batch_t1[..., None]], dim=-1)
        
        
        concat2 = torch.cat((g_x3, g_y), dim=1)
        # regression initial alignment
        out2 = self.decoder_qt(concat2)
        batch_t2, batch_quat2 = out2[:, :3], out2[:, 3:] / (
                torch.norm(out2[:, 3:], dim=1, keepdim=True) + 1e-8)
        batch_R2 = batch_quat2mat(batch_quat2)
        batch_T2 = torch.cat([batch_R2, batch_t2[..., None]], dim=-1)
        
        
        return batch_T,batch_T1,batch_T2,x_ol, y_ol