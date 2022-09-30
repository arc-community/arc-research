# original source: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# slightly adapted for ARC ViT experiments


import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, out_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, default(out_dim, dim)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        tie_ff_weights=False,
        tie_attn_weights=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        ff = lambda: PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        attn = lambda: PreNorm(
            dim,
            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
        )

        if tie_ff_weights:
            shared_ff = ff()
            ff = lambda: shared_ff
        if tie_attn_weights:
            shared_attn = attn()
            attn = lambda: shared_attn

        for _ in range(depth):
            self.layers.append(nn.ModuleList([ff(), attn()]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT_NoEmbed(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        patch_dim,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        tie_ff_weights=False,
        tie_attn_weights=False
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_dim = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            tie_ff_weights,
            tie_attn_weights,
        )

        self.pool = pool
        # self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x, y=None, raw=False):
        x = self.to_dim(x)

        if y is not None:
            x = torch.cat((y, x), dim=1)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if raw:
            return x

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        # x = self.to_latent(x)
        return self.mlp_head(x)


class EncDecViT(nn.Module):
    def __init__(
        self,
        *,
        num_patches,  # number of training examples
        patch_dim,  # C*H*W
        num_classes,  # number of output elements, e.g. pixels * channels
        dim,
        enc_depth,
        dec_depth,
        heads,
        mlp_dim,
        dim_head=64,
        tie_ff_weights=False,
        tie_attn_weights=False,
        num_latents=1,
        latent_dim=None,
        cross_heads=None,
        cross_dim_head=None
    ):
        super().__init__()

        self.encoder = ViT_NoEmbed(
            num_patches=num_patches,
            patch_dim=patch_dim,
            num_classes=num_classes,
            dim=dim,
            depth=enc_depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            tie_ff_weights=tie_ff_weights,
            tie_attn_weights=tie_attn_weights,
        )

        latent_dim = default(latent_dim, dim)
        cross_heads = default(cross_heads, heads)
        cross_dim_head = default(cross_dim_head, dim_head)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    CrossAttention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim, hidden_dim=latent_dim * 4, out_dim=dim)),
            ]
        )

        self.decoder = ViT_NoEmbed(
            num_patches=num_latents + 1,  # description token + test input
            patch_dim=patch_dim,
            num_classes=num_classes,
            dim=dim,
            depth=dec_depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            tie_ff_weights=tie_ff_weights,
            tie_attn_weights=tie_attn_weights,
        )

    def forward(self, train_examples, test_example):
        context = self.encoder.forward(train_examples, raw=True)
        cross_attn, cross_ff = self.cross_attend_blocks

        # query 'transformation token' from encoder output
        x = repeat(self.latents, "n d -> b n d", b=train_examples.size(0))
        x = cross_attn(x, context=context, mask=None) + x
        x = cross_ff(x)

        # combine transformation token and test-example and pass them through decoder
        return self.decoder(test_example, y=x)
