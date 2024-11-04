# ChatIITK - An Advanced RAG ChatBot for IITK Junta

Developed by BCS under its Summer Project - Lluminating Language

## Environment Setup

The app is still in prototyping stage so before running, create a separate virtual environment, so that there are no conflicts.
```
conda create -n ChatIITK python=3.10.0
conda activate ChatIITK
```

After successfully creating a virtual env, install all the requirement for the project

```
pip install -r requirements.txt
```

## Inference
Before start inferencing, we need to ingest our data into our VectorDB(ChromaDB in this case).

You'll need a Google Gemini API key to run the model. You can get one from the Google AI Studio (https://makersuite.google.com/app/apikey).

You can provide the API key in two ways:
1. Environment variable: Set GEMINI_API_KEY in your .env file or environment
2. Command line flag: Use --gemini_api_key when running the script

```
python ingest.py --device_type cuda 
```
After the ingestion of data is complete you can see, local vectorDB files in the `DB` folder, Now:

- To start the Terminal interface run `python run_ChatIITK.py`
    - Additional Flags with `run_ChatIITK.py`
    1. `--gemini_api_key YOUR_GEMINI_API_KEY` : Your Google Gemini API key for model inference.
    2. `--save_qa` : You can store user question and model responsesinto a csv file. This file will be stored as `/local_chat_history/qa_log.csv` 
    3. `--use_history or -h` : You can enable chat history. This is disable by default. You can enable it by using the `--use_history or -h` flag.
    4. `--show_sources` : To show, which chunks are being retrieved from your retriever. By default, it will show 4 different sources/chunks. You can change the number of sources/chunks
    5. `--help` : To get help on these flags.


- To start the Streamlit UI run `streamlit run ChatIITK_UI.py`

## How to select Different Embedding model and LLM:

- To select different model, you can specify them in `constants.py` 
- Change the `MODEL_ID` and `MODEL_BASENAME`. If you are using a quantized model (GGML, GPTQ, GGUF), you will need to provide `MODEL_BASENAME`. For unquantized models, set `MODEL_BASENAME` to `NONE`
- Refer to the examples in the `constant.py` file and check the HuggingFace Hub for more models.


