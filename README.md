# My RAG

This is a personal project where I have implemented a RAG using the LangChain 
and LangGraph frameworks, and the OpenAI API. You can use a local LLM using 
Ollama.

I have used the Streamlit framework for the chatbot frontend implementation.

## Usage

You need to set your `OPENAI_API_KEY` env var, it can be done through a `.env`
file in the root directory. You also need to create a `data` dir in the root 
directory with your `.md` files with your documents in markdown.

It uses the `google-cloud-logging` package for log input and outputs, so you
need to have your GCP project configured via GCP or specify a path for 
a google application credetials .json file through the `GOOGLE_APPLICATION_CREDENTIALS` 
env var.

You can run it via `Docker`:
```bash
docker compose build
docker compose up
```
or installing the requirments and runing `streamlit run main.py`

## Project structure

- `agent/RAG.py` contains the RAG class with all nodes and edges.
- `utils/logger.py` contains the logger configuration for GCP.
- `main.py` contains the chatbot logic and the streamlit config.

