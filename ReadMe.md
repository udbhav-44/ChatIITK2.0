# ChatIITK - RAG-based Chatbot for IITK Community

A Retrieval-Augmented Generation (RAG) chatbot developed by BCS under its Summer Project - Lluminating Language. The chatbot helps IITK students access information about campus life, academics, and policies.

## Project Structure

```
ChatIITK2.0/
├── SOURCE_DOCUMENTS/         # Knowledge base documents
│   ├── Constitution.md
│   ├── UG-Manual.pdf 
│   ├── Gymkhana_website.txt
│   └── ...
├── DB/                      # ChromaDB vector store
├── retrieval/               # Advanced retrieval components
│   ├── advanced_retrieval.py
│   └── web_search.py
├── models/                  # Model weights and configurations
├── ChatIITK_chainlit.py    # Chainlit UI implementation
├── ChatIITK_UI.py          # Streamlit UI implementation
└── constants.py            # Configuration constants
```

## Setup

1. Create a virtual environment:
```bash
conda create -n ChatIITK python=3.10.0
conda activate ChatIITK
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
- Get a Groq API key from [Groq](https://console.groq.com)
- Add to `.env` file or set as environment variable:
```
GROQ_API_KEY=your_key_here
```

4. Ingest documents into the vector database:
```bash
python ingest.py --device_type cuda
```

## Usage

### Chainlit UI
```bash
python ChatIITK_chainlit.py
```

### Streamlit UI
```bash
streamlit run ChatIITK_UI.py
```

### Terminal Interface
```bash
python run_ChatIITK.py [options]
```

Options:
- `--save_qa`: Store Q&A pairs in CSV at `/local_chat_history/qa_log.csv`
- `--use_history` or `-h`: Enable chat history (disabled by default)
- `--show_sources`: Show retrieved document chunks (default: 4 sources)

## Configuration

Model and embedding settings can be configured in `constants.py`:

- Change `MODEL_ID` and `MODEL_BASENAME` for different LLM models
- Modify `EMBEDDING_MODEL_NAME` for different embedding models
- Adjust retrieval parameters in `retrieval/advanced_retrieval.py`

## Contributing

This project is maintained by BCS (Brain and Cognitive Society) at IIT Kanpur. For questions or contributions, please contact the maintainers.

