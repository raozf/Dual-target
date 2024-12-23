#改初始化变换，work_dirs,feat_dim=256,改初始变换,共享卷积核参数
"work_dirs_head=1,1240epoches,初始最终权重均为1.2"
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import math
from sklearn.decomposition import PCA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat , batch_transform
from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=16,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
    
    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
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
def z_order_sorting(points):
    def compute_z_order(point):
        z_order = 0
        x, y, z = (point * 1000).astype(int)
        for i in range(10):
            z_order |= ((x >> i) & 1) << (3 * i + 2)
            z_order |= ((y >> i) & 1) << (3 * i + 1)
            z_order |= ((z >> i) & 1) << (3 * i)
        return z_order
    z_orders = np.apply_along_axis(compute_z_order, 1, points)
    sorted_indices = np.argsort(z_orders)
    return sorted_indices

def process_point_clouds(point_clouds):
    pca = PCA(n_components=3)
    sorted_batches = []
    for batch in point_clouds:
        batch_cpu = batch.cpu().detach().numpy()
        reduced_points = pca.fit_transform(batch_cpu)
        sorted_indices = z_order_sorting(reduced_points)
        # 使用排序索引对原始批次进行排序
        sorted_batch = batch[sorted_indices]
        sorted_batches.append(sorted_batch)
    return torch.stack(sorted_batches)    
class PointNet1(nn.Module):
    def __init__(self, in_dim,gn):
        super(PointNet1 , self).__init__()
#        self.weights_a = nn.Parameter(torch.randn(1))
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
        self.mamba = Mamba2(1472)
    def forward(self, x, pooling=True):
        point1_feat64 = self.conv_block1(x)
#        expanded_weights_a = self.weights_a.unsqueeze(0).unsqueeze(2)  # 从1变为(1, 64, 1)
#        weipoint1_feat64 = self.weights_a * point1_feat64
        point2_feat64 = self.conv_block2(point1_feat64)
#        expanded_weights_b = self.weights_b.unsqueeze(0).unsqueeze(2)  # 从1变为(1, 64, 1)
#        weipoint2_feat64 = self.weights_a * point2_feat64
        
        point3_feat64 = self.conv_block3(point2_feat64)
#        expanded_weights_c = self.weights_c.unsqueeze(0).unsqueeze(2)  # 从1变为(1, 64, 1)
#        weipoint3_feat64 = self.weights_a * point3_feat64
        
        x = self.conv_block4(point3_feat64)
        point_feat1024 = self.conv_block5(x)
#        L = torch.cat([point1_feat64,point2_feat64,point3_feat64,point_feat1024], dim = 1)
        L = torch.cat([point1_feat64, point2_feat64, point3_feat64,x, point_feat1024], dim = 1)
        L = process_point_clouds(L)
        L = L + self.mamba(L.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        if not pooling:
            return L
        x ,_ = torch.max(L, dim=2)
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
    
class Coarse2(nn.Module):
    def __init__(self, in_dim, gn):
        super(Coarse2, self).__init__()
#
        self.encoder1 = PointNet1(in_dim=in_dim,gn=gn)
        
        self.decoder_ol = PointNet_de(in_dim=5888,
                                   gn=gn,
                                   out_dims=[2048,2048,512,256,2])
#       [2048, 2048,512, 256, 2]
        self.decoder_qt = MLPs(in_dim=2944,
                               gn = gn,
                               mlps=[1024,1024,256,256,7])
#        self.pointer = Transformer()
    def forward(self, src, tgt):
        '''
        Context-Guided Model for initial alignment and overlap score.
        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :return: T0: (B, 3, 4), OX: (B, N, 2), OY: (B, M, 2)
        '''
        x = src.permute(0, 2, 1).contiguous()
        y = tgt.permute(0, 2, 1).contiguous()
        f_x, g_x = self.encoder1(x)
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
#        ff_x , ff_y = self.pointer(f_x1,f_y)                                         
        g_x_expand = torch.unsqueeze(g_x1, dim=-1).expand_as(f_x1)
        g_y_expand = torch.unsqueeze(g_y, dim=-1).expand_as(f_y)
        f_x_ensemble = torch.cat([f_x1, g_x_expand, g_y_expand, g_x_expand - g_y_expand], dim=1)
        f_y_ensemble = torch.cat([f_y, g_y_expand, g_x_expand, g_y_expand - g_x_expand], dim=1)
        x_ol = self.decoder_ol(f_x_ensemble, pooling=False)
        y_ol = self.decoder_ol(f_y_ensemble, pooling=False)
        return batch_T, x_ol, y_ol