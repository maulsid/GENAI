# GENAI Agent - Eval Runner

Minimal repo to run simple QA evaluations using Groq + LangSmith.

## Prerequisites
- macOS
- Python 3.10+
- Git

## Setup
1. Create and activate a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate

2. Install dependencies (adjust as needed):
   - pip install python-dotenv langchain-groq langsmith numpy

3. Create a `.env` file in the project root with your Groq key:
   - GROQ_API_KEY=your_groq_api_key_here

## Run evaluation
From project root:
- python evals_func/run_eval.py

The script uses `evals_func/run_eval.py` which loads `.env`, constructs a ChatGroq LLM, then calls `client.evaluate(...)`. Output is printed to stdout.

## Project layout
- evals_func/run_eval.py — main evaluation entrypoint
- evals_func/utils.py — helper metrics (cosine_similarity, exact_match)
- .env — environment variables (not checked in)

## Notes
- Ensure `GROQ_API_KEY` is set.
- Adjust model or evaluator list in `run_eval.py` as needed.

```// filepath: /Users/mauli/maulidata/projects/GENAI/agent/README.md
# GENAI Agent - Eval Runner

Minimal repo to run simple QA evaluations using Groq + LangSmith.

## Prerequisites
- macOS
- Python 3.10+
- Git

## Setup
1. Create and activate a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate

2. Install dependencies (adjust as needed):
   - pip install python-dotenv langchain-groq langsmith numpy

3. Create a `.env` file in the project root with your Groq key:
   - GROQ_API_KEY=your_groq_api_key_here

## Run evaluation
From project root:
- python evals_func/run_eval.py

The script uses `evals_func/run_eval.py` which loads `.env`, constructs a ChatGroq LLM, then calls `client.evaluate(...)`. Output is printed to stdout.

## Project layout
- evals_func/run_eval.py — main evaluation entrypoint
- evals_func/utils.py — helper metrics (cosine_similarity, exact_match)
- .env — environment variables (not checked in)

## Notes
- Ensure `GROQ_API_KEY` is set.
- Adjust model or evaluator list in `run_eval.py` as needed.
