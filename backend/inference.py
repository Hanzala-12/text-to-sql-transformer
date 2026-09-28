"""Model definition and decoding logic for serving the trained Text-to-SQL
Transformer. Mirrors Text_to_SQL_Transformer.ipynb exactly (same classes,
same config) so a checkpoint trained there loads here without changes.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import sentencepiece as spm

# ---------------------------------------------------------------- constants
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<"]
MAX_COLS = 64

D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 1024
DROPOUT = 0.1


# ---------------------------------------------------------------- starter code (embeddings)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def forward(self, ids):
        return self.emb(ids) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class InputLayer(nn.Module):
    def __init__(self, token_emb, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.tok = token_emb
        self.pos = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, ids):
        return self.pos(self.tok(ids))


# ---------------------------------------------------------------- attention
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, L, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * d_k)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        out = self._merge_heads(out)
        return self.w_o(self.dropout(out)), attn


# ---------------------------------------------------------------- layers
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.dropout(sublayer_out))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, src_mask):
        attn_out, attn_w = self.self_attn(x, x, x, src_mask)
        x = self.add_norm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)
        return x, attn_w


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_out, tgt_mask, cross_mask):
        self_out, self_attn_w = self.self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, self_out)
        cross_out, cross_attn_w = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.add_norm2(x, cross_out)
        ffn_out = self.ffn(x)
        x = self.add_norm3(x, ffn_out)
        return x, self_attn_w, cross_attn_w


# ---------------------------------------------------------------- masks + full model
def make_padding_mask(ids, pad_id=0):
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, src_mask)
            attn_weights.append(attn_w)
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, tgt_mask, cross_mask):
        self_attns, cross_attns = [], []
        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_out, tgt_mask, cross_mask)
            self_attns.append(self_attn_w)
            cross_attns.append(cross_attn_w)
        return x, self_attns, cross_attns


class Transformer(nn.Module):
    def __init__(self, enc_input_layer, dec_input_layer, shared_token_emb,
                 vocab_size, d_model=256, num_heads=4, num_enc_layers=3,
                 num_dec_layers=3, d_ff=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.enc_input = enc_input_layer
        self.dec_input = dec_input_layer
        self.encoder = Encoder(num_enc_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_dec_layers, d_model, num_heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        self.generator.weight = shared_token_emb.emb.weight

    def encode(self, src):
        src_mask = make_padding_mask(src, self.pad_id)
        x = self.enc_input(src)
        enc_out, _ = self.encoder(x, src_mask)
        return enc_out, src_mask

    def decode(self, tgt_in, enc_out, src_mask):
        B, T = tgt_in.shape
        pad_mask = make_padding_mask(tgt_in, self.pad_id)
        causal_mask = make_causal_mask(T, tgt_in.device)
        tgt_mask = pad_mask & causal_mask
        x = self.dec_input(tgt_in)
        dec_out, self_attns, cross_attns = self.decoder(x, enc_out, tgt_mask, src_mask)
        return dec_out, self_attns, cross_attns

    def forward(self, src, tgt_in):
        enc_out, src_mask = self.encode(src)
        dec_out, self_attns, cross_attns = self.decode(tgt_in, enc_out, src_mask)
        logits = self.generator(dec_out)
        return logits, cross_attns

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size, device, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT):
    shared_emb = TokenEmbedding(vocab_size, d_model, PAD_ID)
    enc_input = InputLayer(shared_emb, d_model, dropout=dropout)
    dec_input = InputLayer(shared_emb, d_model, dropout=dropout)
    model = Transformer(enc_input, dec_input, shared_emb, vocab_size,
                         d_model=d_model, num_heads=num_heads,
                         num_enc_layers=num_layers, num_dec_layers=num_layers,
                         d_ff=d_ff, dropout=dropout, pad_id=PAD_ID)
    return model.to(device)


# ---------------------------------------------------------------- data prep helpers
def encode_source(question, header):
    cols = " ".join(f"<c{i}> {name}" for i, name in enumerate(header))
    return f"{question.strip()} <sep> {cols}".lower()


# ---------------------------------------------------------------- decoding
@torch.no_grad()
def greedy_decode(model, src, max_len=64, device="cpu"):
    model.eval()
    src = src.to(device)
    enc_out, src_mask = model.encode(src)
    B = src.size(0)
    ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    cross_attns = None
    for _ in range(max_len):
        dec_out, _, cross_attns = model.decode(ys, enc_out, src_mask)
        logits = model.generator(dec_out[:, -1])
        next_tok = logits.argmax(-1, keepdim=True)
        next_tok = torch.where(finished.unsqueeze(-1), torch.full_like(next_tok, EOS_ID), next_tok)
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(-1) == EOS_ID)
        if finished.all():
            break
    return ys, cross_attns


@torch.no_grad()
def beam_search_decode(model, src, beam_size=4, max_len=64, device="cpu"):
    model.eval()
    src = src.to(device)
    enc_out, src_mask = model.encode(src)
    beams = [(torch.tensor([[BOS_ID]], device=device), 0.0, False)]
    for _ in range(max_len):
        candidates = []
        for ys, score, done in beams:
            if done:
                candidates.append((ys, score, done))
                continue
            dec_out, _, _ = model.decode(ys, enc_out, src_mask)
            logits = model.generator(dec_out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk_logp, topk_idx = log_probs.topk(min(beam_size, log_probs.size(-1)))
            for lp, idx in zip(topk_logp, topk_idx):
                new_ys = torch.cat([ys, idx.view(1, 1)], dim=1)
                candidates.append((new_ys, score + lp.item(), idx.item() == EOS_ID))
        candidates.sort(key=lambda c: c[1] / c[0].size(1), reverse=True)
        beams = candidates[:beam_size]
        if all(done for _, _, done in beams):
            break
    best_ys = beams[0][0]
    _, _, cross_attns = model.decode(best_ys, enc_out, src_mask)
    return best_ys, cross_attns


def parse_prediction(text):
    try:
        tokens = text.strip().split()
        assert tokens and tokens[0] == "select"
        i = 1
        agg = 0
        agg_names = [a.lower() for a in AGG_OPS if a]
        if tokens[i] in agg_names:
            agg = [a.lower() for a in AGG_OPS].index(tokens[i])
            i += 1
        sel_tok = tokens[i]
        assert sel_tok.startswith("<c") and sel_tok.endswith(">")
        sel = int(sel_tok[2:-1])
        i += 1
        conds = []
        while i < len(tokens):
            assert tokens[i] in ("where", "and")
            i += 1
            col_tok = tokens[i]
            assert col_tok.startswith("<c") and col_tok.endswith(">")
            col = int(col_tok[2:-1])
            i += 1
            op = tokens[i]
            assert op in COND_OPS
            i += 1
            val_tokens = []
            while i < len(tokens) and tokens[i] != "and":
                val_tokens.append(tokens[i])
                i += 1
            value = " ".join(val_tokens)
            assert value != ""
            conds.append([col, COND_OPS.index(op), value])
        return {"sel": sel, "agg": agg, "conds": conds}
    except (AssertionError, IndexError, ValueError):
        return None


def readable_sql(parsed, header):
    agg = AGG_OPS[parsed["agg"]]
    col = header[parsed["sel"]] if parsed["sel"] < len(header) else f"<c{parsed['sel']}>"
    sel_expr = f"{agg}({col})" if agg else col
    sql = f"SELECT {sel_expr} FROM table"
    if parsed["conds"]:
        clauses = []
        for c, op, val in parsed["conds"]:
            col_name = header[c] if c < len(header) else f"<c{c}>"
            clauses.append(f"{col_name} {COND_OPS[op]} '{val}'")
        sql += " WHERE " + " AND ".join(clauses)
    return sql


def _strip_special(ids):
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    return ids


def decode_example(model, sp, src_row, method="greedy", beam_size=4, max_len=64, device="cpu"):
    if method == "greedy":
        ys, cross_attns = greedy_decode(model, src_row, max_len=max_len, device=device)
    else:
        ys, cross_attns = beam_search_decode(model, src_row, beam_size=beam_size, max_len=max_len, device=device)
    ids = _strip_special(ys[0].tolist())
    text = sp.decode(ids)
    parsed = parse_prediction(text)
    prediction = {"error": "parse"} if parsed is None else {"query": parsed}
    return prediction, text


# ---------------------------------------------------------------- loading + one-shot inference
class TextToSQLEngine:
    """Loads the checkpoint + tokenizer once, then answers queries."""

    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        vocab_size = ckpt.get("vocab_size", self.sp.get_piece_size())
        self.model = build_model(vocab_size, self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.epoch = ckpt.get("epoch")
        self.dev_loss = ckpt.get("dev_loss")

    def query(self, question: str, header: list[str], method: str = "beam", beam_size: int = 4, max_len: int = 64):
        if not header:
            raise ValueError("At least one column name is required.")
        src_text = encode_source(question, header)
        src_ids = self.sp.encode(src_text) + [EOS_ID]
        src = torch.tensor([src_ids], dtype=torch.long)

        prediction, tokenized = decode_example(
            self.model, self.sp, src, method=method, beam_size=beam_size,
            max_len=max_len, device=self.device,
        )

        result = {
            "tokenized": tokenized,
            "source": src_text,
        }
        if "query" in prediction:
            result["sql"] = readable_sql(prediction["query"], header)
            result["parsed"] = prediction["query"]
            result["ok"] = True
        else:
            result["ok"] = False
            result["error"] = "The model's output could not be parsed into a valid query."
        return result
