import torch
from torch import nn

from model.decoder_block import TransformerDecoderBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.decoder_blocks = nn.Sequential(
            *[TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.emb_dim)
        self.out = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.decoder_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits