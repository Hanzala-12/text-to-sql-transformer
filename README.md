# Text-to-SQL with the Transformer

Generative AI - Assignment 02. An encoder-decoder Transformer (Vaswani et
al., 2017), implemented from scratch, that turns an English question over a
table into a SQL query, trained on WikiSQL.

## Contents

```
Text_to_SQL_Transformer.ipynb   standalone notebook - starter code + Tasks 1-5, run on Kaggle/Colab
results/                        output of the notebook run (checkpoint, plots, predictions, samples.md)
backend/                        FastAPI server that loads results/best.pt and serves the model
frontend/                       React front end (Task 6.1) that calls the backend
README.md                       this file
```

Everything is in the one notebook - the starter code and the full Task 2-5
implementation are written directly in its cells, with no other file
required. Nothing in it uses `nn.Transformer`, `nn.TransformerEncoder/DecoderLayer`,
`nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging Face
`transformers`, or any pretrained weights - only `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout`, `nn.ReLU`, `torch.softmax`, `torch.matmul`, and
the given starter code.

## Run it

Upload `Text_to_SQL_Transformer.ipynb` to Kaggle (or Colab):

1. Settings -> Accelerator: **GPU**, Internet: **On**.
2. Run all cells. It clones and extracts WikiSQL itself; `VOCAB_SIZE`,
   `BATCH_SIZE` and `NUM_EPOCHS` switch to the assignment's real values
   (8000 / 64 / 20) automatically once real data is found. If no real data is
   found it falls back to a tiny synthetic dataset with the same schema, so
   the notebook is still runnable as a smoke test.
3. The last cells run WikiSQL's own `evaluate.py` on dev (greedy + beam) and
   once on test, and zip everything in `results/` into `results_bundle.zip`
   for download from the notebook's Output tab.

## Configuration (fixed, per the assignment)

| | |
|---|---|
| d_model | 256 |
| heads | 4 (dk = dv = 64) |
| encoder / decoder layers | 3 / 3 |
| feed-forward inner size | 1024 |
| dropout | 0.1 |
| normalisation | post-norm, `LayerNorm(x + Sublayer(x))` |
| weight sharing | encoder embedding = decoder embedding = output projection |
| label smoothing | 0.1 |
| optimiser | Adam, beta1=0.9, beta2=0.98, eps=1e-9 |
| LR schedule | Noam (paper Eq. 3), warmup=4000 |
| batch size / epochs | 64 / 20 |

## Correctness checks (run inside the notebook, on random tensors)

- **Causal mask**: changing the last token of a decoder input leaves every
  earlier decoder output unchanged.
- **Padding mask**: appending extra `<pad>` tokens to the source leaves the
  decoder output unchanged; cross-attention rows sum to 1 over the unmasked
  positions, with ~0 attention mass on the padded ones.
- **Weight sharing**: `model.generator.weight is model.enc_input.tok.emb.weight`
  (an `is` check, not just equal values).
- **Learning-rate schedule**: plotted for the first 20,000 steps; rises
  linearly for 4,000 steps then decays as `step^-0.5`.
- **Gold round-trip**: gold dev targets, parsed and run through WikiSQL's own
  `evaluate.py` - execution accuracy must be above 99%.

## Results

*Fill in after the real 20-epoch run on the full WikiSQL data.*

### Table 1 - Data

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56,355 | 8,421 | 15,878 |
| Mean / max source length (tokens) | | | |
| Mean / max target length (tokens) | | | |
| Pairs dropped as too long | | – | – |

### Table 2 - Model and training

| | |
|---|---|
| Trainable parameters | |
| Epochs trained / best epoch | |
| Best dev loss | |
| Training time and GPU | |

### Table 3 - Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | | | |
| Dev | beam (4) | | | |
| Test | | | | |

### Table 4 - Component accuracy (dev)

| sel column correct (%) | agg correct (%) | WHERE clause correct (%) |
|---|---|---|
| | | |

### Figures

- `results/positional_encoding.png` - positional-encoding heat-map (Task 1.4).
- `results/loss_curve.png` - training and dev loss per epoch.
- `results/lr_schedule.png` - learning-rate schedule.
- `results/cross_attention.png` - cross-attention map for one dev example (Task 5.3).

### Qualitative samples

`results/samples.md` - five correct, five wrong dev examples, with the
failure named for each wrong one.

## Web front end (Task 6.1)

A React app talking to a small FastAPI server that loads the trained
checkpoint and runs real inference - not a mock.

1. **Get the checkpoint in place.** Unzip `results_bundle.zip` from the
   Kaggle run so that `results/best.pt` and `results/sql_sp.model` exist at
   the repo root (already handled if you just extract the zip into
   `results/`).

2. **Start the backend** (dependencies are already installed in
   `backend/.venv`):
   ```bash
   cd backend
   .venv\Scripts\uvicorn server:app --port 8000        # Windows
   .venv/bin/uvicorn server:app --port 8000             # macOS/Linux
   ```
   Check it loaded correctly: `curl http://localhost:8000/api/status`.

3. **Start the frontend** (dependencies already installed in
   `frontend/node_modules`):
   ```bash
   cd frontend
   npm run dev
   ```
   Open the printed `http://localhost:5173` URL.

Type a question and a comma-separated list of column names, pick greedy or
beam decoding, and it calls the real model and shows the generated SQL (with
the raw tokenised output available underneath). Screenshot this for the
README once real results exist.

## Not included here

Blog post, LinkedIn post, and the GitHub push itself - out of scope here by
design.
