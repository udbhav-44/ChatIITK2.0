import os
import logging
import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import torch
import utils
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma


from langchain_groq import ChatGroq

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)


def load_model(api_key, LOGGING=logging):
    """
    Initialize the Groq model using the provided API key.
    
    Args:
        api_key (str): The API key for accessing Groq.
        LOGGING (logging.Logger): Logger instance for logging messages.
        
    Returns:
        ChatGroq: A LangChain wrapper for the Groq model.
    """
    logging.info("Initializing Groq model")
    
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768"
    )
    
    logging.info("Groq model initialized")

    return llm


def retrieval_qa_pipline(gemini_api_key, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings("cpu")  # Use CPU for embeddings since we're using cloud API for inference

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(gemini_api_key, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        # Initialize advanced retriever
        from advanced_retrieval import AdvancedRetriever
        advanced_retriever = AdvancedRetriever(db, embeddings, llm)
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=advanced_retriever.compression_retriever,  # Use advanced retriever
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--groq_api_key",
    required=False,
    help="Groq API key (can also be set via GROQ_API_KEY environment variable)",
    default=lambda: os.environ.get("GROQ_API_KEY", None),
)

@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(groq_api_key, show_sources, use_history, model_type, save_qa):
    if not groq_api_key:
        raise click.UsageError("Groq API key must be provided either via --groq_api_key flag or GROQ_API_KEY environment variable")
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info("Using Groq API for inference")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(groq_api_key, use_history, promptTemplate_type=model_type)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
