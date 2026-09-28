# Text-to-SQL Backend

FastAPI server that loads the trained Transformer checkpoint and serves
predictions to the React front end.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

## Run

Place `best.pt` and `sql_sp.model` (produced by the training notebook) in
`../results/`, then:

```bash
uvicorn server:app --port 8000
```

`GET /api/status` reports whether the checkpoint loaded. `POST /api/query`
takes `{"question": str, "columns": [str], "method": "greedy" | "beam"}` and
returns the generated SQL.

## Structure

```
inference.py   model definition and decoding, mirrors the training notebook
server.py      FastAPI app and endpoints
```
