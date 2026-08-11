# LLM-based Ecommerce Chatbot

A semantic-routing chatbot for ecommerce support that combines:
- **Semantic Router** — classifies incoming user queries into intents (FAQ, product search, chitchat, code help)
- **RAG over FAQs** — ChromaDB + local embeddings for FAQ question answering
- **Natural-language product search** — Groq LLM converts plain-English product questions into SQL against a local product database, then converts results back into natural language

All embeddings and the vector store run **locally and free** (no OpenAI, no paid embedding API). The only paid/external dependency is the Groq LLM API, which has a generous free tier.

---

## Project structure

```
.
├── config.py            # All constants (paths, model names, table/collection names)
├── router.py             # Semantic Router — classifies query into an intent/route
├── main.py                # Entry point — routes query, dispatches to the right handler
├── faq_router.py           # FAQ RAG pipeline (ChromaDB + Groq) — builds index from faq.csv, answers FAQ questions
├── sql.py                    # NL-to-SQL pipeline — answers product questions against products.db
├── csv_to_sqlite.py           # Loads a product CSV into a SQLite database
├── .env.example                 # Template for required environment variables
├── .gitignore
└── notebooks/
    ├── amazon_scraper.ipynb        # Selenium scraper (optional — use if you don't have a product CSV)
    └── amazon_kaggle_dataset.ipynb  # Downloads a public product dataset from Kaggle (no scraping)
```

---

## How it works

1. **`main.py`** takes the user's query and calls `router.py`.
2. **`router.py`** (Semantic Router + local `FastEmbedEncoder`) classifies the query into one of: `faq`, `product_search`, `chitchat`, `code_help`.
3. Based on the matched route, `main.py` dispatches to:
   - **`faq_router.py`** → `faq_chain()` — retrieves relevant FAQ answers from ChromaDB, generates a grounded answer via Groq.
   - **`sql.py`** → `sql_chain()` — generates a SQL query via Groq (from `sql_prompt`), runs it against `products.db`, then converts the result rows into a natural-language product list via `comprehension_prompt`.
   - Other routes get simple canned responses.

---

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install chromadb sentence-transformers pandas semantic-router groq python-dotenv
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Then edit `.env` and add your Groq API key (free tier available at [console.groq.com](https://console.groq.com)):
```
GROQ_API_KEY=your-actual-key-here
```

### 4. Prepare your data

**FAQ data:** place a `faq.csv` in the project root with `question` and `answer` columns. The index is built automatically (and rebuilt only when the file changes) the first time `faq_chain()` runs.

**Product data:** get a product CSV (e.g. via `notebooks/amazon_kaggle_dataset.ipynb`, or your own `yoga.csv`), then load it into SQLite:
```bash
python csv_to_sqlite.py
```
This creates `products.db` with a `products` table. Update `CSV_PATH` in `csv_to_sqlite.py` to point at your actual file first.

Expected product columns (used by `sql.py`'s schema prompt):
```
name, main_category, sub_category, image, link, ratings, no_of_ratings, discount_price, actual_price
```
If your columns differ, update the `<schema>` block in `sql.py`'s `sql_prompt` to match.

### 5. Run
```bash
python main.py
```

---

## Key design notes

- **Embeddings are 100% local and free** — via `sentence-transformers` (`BAAI/bge-small-en-v1.5`), downloaded once and cached; no API key, no internet needed after first run.
- **Groq is used only for text generation** — SQL generation, FAQ answer generation, and result-to-natural-language comprehension. It has a free tier suitable for development.
- **FAQ index syncing is hash-based** — `faq_router.py` only re-embeds `faq.csv` when its contents actually change, tracked via `chroma_store/faq_hash.json`.
- **SQL safety guard** — `sql.py` only allows single `SELECT` statements; anything else (INSERT/UPDATE/DELETE/DROP/etc.) is rejected before execution.
- **Keyword search over full-phrase search** — the SQL-generation prompt explicitly instructs the model to break multi-word queries into individual keywords ANDed together (each checked across `name` OR `sub_category`), rather than hedging with multiple overly-strict full-phrase `LIKE` variants.

