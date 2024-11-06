import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import logging
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from prompt_template_utils import get_prompt_template
from utils import get_embeddings
from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS,
)
from langchain_groq import ChatGroq
from retrieval import advanced_retrieval

# Load environment variables
load_dotenv()

def load_model(api_key, LOGGING=logging):
    """
    Initialize the Groq model using the provided API key.
    """
    logging.info("Initializing Groq model")
    
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=api_key,
        model_name="llama-3.2-11b-text-preview"
    )
    
    logging.info("Groq model initialized")
    return llm

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        await cl.Message(content="Error: Please set GROQ_API_KEY environment variable").send()
        return

    # Load embeddings and vectorstore
    embeddings = get_embeddings("cpu")
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # Initialize the model and QA chain
    llm = load_model(groq_api_key)
    prompt, memory = get_prompt_template(promptTemplate_type="llama3", history=True)
    
    # Initialize advanced retriever
    logging.info("Initializing retriever...")
    advanced_retriever = advanced_retrieval.AdvancedRetriever(db, embeddings, llm)
    logging.info("Retriever initialized successfully")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=advanced_retriever.compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )

    # Store the chain in the user session
    cl.user_session.set("qa_chain", qa_chain)

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    qa_chain = cl.user_session.get("qa_chain")
    
    # Create a Chainlit response message
    response = await cl.make_async(qa_chain)(message.content)
    answer, docs = response["result"], response["source_documents"]

    # Send the answer
    await cl.Message(content=answer).send()

    # Send the sources
    source_docs = []
    for i, doc in enumerate(docs, 1):
        source_text = f"Source {i}:\n{doc.metadata['source']}\n{doc.page_content}\n"
        source_docs.append(source_text)
    
    if source_docs:
        sources_combined = "\n".join(source_docs)
        await cl.Message(content=sources_combined, author="Sources").send()

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", 
        level=logging.INFO
    )
