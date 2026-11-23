"""RGB denoiser with configurable transformer depth."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def patchify(rgb, patch):
    """BCHW RGB -> row-major tokens, with channels last inside each patch."""
    b, channels, height, width = rgb.shape
    return rgb.permute(0, 2, 3, 1).reshape(b, height // patch, patch, width // patch, patch, channels).permute(
        0, 1, 3, 2, 4, 5).reshape(b, height * width // patch**2, patch**2 * channels)


def unpatchify(tokens, resolution, patch):
    grid = resolution // patch
    return tokens.reshape(-1, grid, grid, patch, patch, 3).permute(0, 1, 3, 2, 4, 5).reshape(
        -1, resolution, resolution, 3).permute(0, 3, 1, 2)


def rotate(value, cosine, sine):
    even, odd = value.unflatten(-1, (-1, 2)).unbind(-1)
    return value * cosine + torch.stack((-odd, even), dim=-1).flatten(-2) * sine


class TimeEmbedding(nn.Module):
    def __init__(self, width, frequency_size):
        super().__init__()
        half = frequency_size // 2
        frequencies = torch.exp(-math.log(10000.) * torch.arange(half).float() / half)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(frequency_size, width), nn.SiLU(), nn.Linear(width, width))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def forward(self, time):
        angle = time.float()[:, None] * self.frequencies
        return self.mlp(torch.cat((angle.cos(), angle.sin()), dim=-1))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg['heads'], cfg['head_dim']
        self.eps = cfg['qk_norm_eps']
        inner = self.heads * self.head_dim
        self.qkv = nn.Linear(cfg['width'], 3 * inner, bias=False)
        self.output = nn.Linear(inner, cfg['width'], bias=False)

    def qk_norm(self, value):
        return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.eps).to(value.dtype)

    def forward(self, value, rotary, identity_attention=False, fused_attention=False):
        batch, count, _ = value.shape
        qkv = self.qkv(value).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, content = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if identity_attention:
            # An identity attention map returns each token's own value vector.
            result = content
        else:
            query = rotate(self.qk_norm(query), *rotary)
            key = rotate(self.qk_norm(key), *rotary)
            if fused_attention:
                result = F.scaled_dot_product_attention(query, key, content)
            else:
                with torch.autocast(value.device.type, enabled=False):
                    scores = (query.float() @ key.float().transpose(-2, -1)) * self.head_dim**-.5
                    weights = scores.softmax(-1)
                result = weights.to(content.dtype) @ content
        return self.output(result.transpose(1, 2).reshape(batch, count, -1))


class SwiGLU(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        inner = round(width * ratio)
        self.input = nn.Linear(width, 2 * inner, bias=False)
        self.output = nn.Linear(inner, width, bias=False)

    def forward(self, value):
        gate, content = self.input(value).chunk(2, -1)
        return self.output(F.silu(gate) * content)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg['width'], eps=cfg['rms_norm_eps'])
        self.attention = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg['width'], eps=cfg['rms_norm_eps'])
        self.mlp = SwiGLU(cfg['width'], cfg['mlp_ratio'])

    def forward(self, value, rotary, identity_attention=False, fused_attention=False):
        value = value + self.attention(self.attention_norm(value), rotary, identity_attention, fused_attention)
        return value + self.mlp(self.mlp_norm(value))


class Denoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if type(cfg['depth']) is not int or cfg['depth'] not in (2, 4, 8, 12, 16):
            raise ValueError('Use a depth of 2, 4, 8, 12 or 16')
        width, patch, resolution = cfg['width'], cfg['patch'], cfg['resolution']
        self.resolution, self.patch = resolution, patch
        self.time = TimeEmbedding(width, cfg['time_frequency_size'])
        self.condition = nn.Sequential(nn.Linear(cfg['attribute_dim'], cfg['condition_width']), nn.SiLU(),
                                       nn.Linear(cfg['condition_width'], width))
        self.patch_in = nn.Linear(patch**2 * 3, width)
        self.slots = nn.Parameter(torch.randn(1, (resolution // patch)**2, width) * .02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['depth'])])
        self.norm = nn.RMSNorm(width, eps=cfg['rms_norm_eps'])
        self.head = nn.Sequential(nn.Linear(width, cfg['head_width']), nn.SiLU(),
                                  nn.Linear(cfg['head_width'], patch**2 * 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        grid = resolution // patch
        y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
        positions = torch.stack((y.flatten(), x.flatten()), -1)
        quarter = cfg['head_dim'] // 4
        frequencies = 10000**(-torch.arange(quarter) / quarter)
        angles = (positions[..., None] * frequencies).repeat_interleave(2, -1).flatten(-2)[None, None]
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())

    def forward(self, xt, time, attributes, *, pag_block=None, fused_attention=False):
        with torch.autocast(xt.device.type, dtype=torch.bfloat16, enabled=xt.is_cuda):
            temporal = self.time(time)
            value = (self.patch_in(xt) + self.slots + temporal[:, None] + self.condition(attributes)[:, None]).to(temporal.dtype)
            rotary = self.rope_cos.to(value.dtype), self.rope_sin.to(value.dtype)
            for index, block in enumerate(self.blocks):
                value = block(value, rotary, identity_attention=(index == pag_block), fused_attention=fused_attention)
            value = self.norm(value)
        with torch.autocast(xt.device.type, enabled=False):
            return unpatchify(self.head(value.float()), self.resolution, self.patch) + .5
