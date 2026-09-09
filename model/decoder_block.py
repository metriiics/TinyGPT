from torch import nn

from model.ffn import FeedForwardNetwork
from model.attention import MaskedMultiHeadAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ffn = FeedForwardNetwork(cfg)
        self.norm1 = nn.LayerNorm(cfg.emb_dim)
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x 
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x 