import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tiktoken
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask=None, cache=None):
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if cache is not None:
            key_cache, value_cache = cache
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            att = att + mask

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)

        out = self.resid_dropout(self.c_proj(out))
        return out, (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )
        self.mlp = MLP(config)

    def forward(self, x, mask=None, cache=None):
        norm = self.ln_1(x)
        att, cache = self.attn(norm, mask=mask, cache=cache)
        x = x + att
        norm = self.ln_2(x)
        mlp = self.mlp(norm)
        x = x + mlp
        return x, cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )

    def _forward_transformer_blocks(
        self, x, pos, mask=None, cache=None, build_cache=False
    ):
        # tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = self.drop(x + pos_emb)
        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.h[i](x, mask=None, cache=cache[i])
        else:
            for block in self.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache

    def _create_causal_mask(self, length: int):
        mask = torch.tril(torch.ones((length, length), dtype=torch.float32))
        return mask.view(1, 1, length, length).to(self.wte.weight.device).bool()

    def _sample_next_token(self, x, temperature):
        logits = x[:, -1:] @ self.wte.weight.T
        y = logits[:, -1, :]
        y = torch.multinomial(
            torch.softmax(y * (1 / temperature), dim=-1), num_samples=1
        )
        return y

    def generate(self, x, visual_embeds, max_new_tokens=256, temperature=0.8):
        # Initial combined embeddings
        text_embeds = self.wte(x)
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=0).unsqueeze(0)

        _, t, _ = combined_embeds.size()
        pos = torch.arange(0, t, dtype=torch.long, device=combined_embeds.device)
        mask = self._create_causal_mask(t)

        combined_embeds, cache = self._forward_transformer_blocks(
            combined_embeds, pos, mask=mask, build_cache=True
        )
        y = self._sample_next_token(combined_embeds, temperature)
        position = t
        yield y

        for _ in range(max_new_tokens):
            position += 1
            x = y
            text_embeds = self.wte(x)
            x, cache = self._forward_transformer_blocks(
                text_embeds,
                torch.tensor([position], dtype=torch.long, device=x.device),
                cache=cache,
            )
            y = self._sample_next_token(text_embeds, temperature)
            yield y

    def forward(self, x, visual_embeds=None, targets=None, padding_mask=None):

        text_embeds = self.wte(x)  # Get text embeddings from token IDs
        combined_embeds = torch.cat(
            [visual_embeds, text_embeds], dim=1
        )  # Concatenate embeddings

        # Generate positional indices for the combined sequence
        batch_size, seq_length, _ = combined_embeds.shape

        assert (
            seq_length <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_length}, block size is only {self.config.block_size}"

        position_ids = (
            torch.arange(0, seq_length, device=combined_embeds.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # Ensure padding_mask is in the correct shape for broadcasting
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_length]

        causal_mask = self._create_causal_mask(seq_length)
        if padding_mask is not None:
            combined_mask = causal_mask & padding_mask
        else:
            combined_mask = causal_mask

        x, _ = self._forward_transformer_blocks(
            combined_embeds, position_ids, combined_mask
        )
        logits = x @ self.wte.weight.T

        # Check if targets exist and compute the loss with ignore_index -100
        if targets is not None:
            vocab_size = logits.size(-1)
            logits = logits.view(
                -1, vocab_size
            )  # Reshape logits to (batch_size * seq_length, vocab_size)
            targets = targets.view(-1)  # Flatten targets to (batch_size * seq_length)

            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            return logits, loss

        return logits, None


def transpose_specific_layers(state_dict):
    layers_to_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key in state_dict.keys():
        if any(key.endswith(suffix) for suffix in layers_to_transpose):
            state_dict[key] = state_dict[key].T
    return state_dict


def generate_text(model: GPT, vision_embeds, tokenizer):
    image_end_token_id = tokenizer.convert_tokens_to_ids("Ġ")

    x = torch.tensor([image_end_token_id])

    tokens = []
    start = time.time()
    for token in model.generate(x, vision_embeds, max_new_tokens=64):
        tok = token.item()
        tokens.append(tok)
        print(tokenizer.decode([tok]), end="", flush=True)
    end = time.time()
    print("---------------")
    print(
        f"time: {end - start:.3f} s, tokens per second: {len(tokens) / (end - start)}"
    )
    print("---------------")


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    state_dict = torch.load("gpt2.bin", map_location="cpu")
    state_dict_transposed = transpose_specific_layers(state_dict)

    model.load_state_dict(state_dict_transposed, strict=False)

    generate_text("Hello my name is", model)
