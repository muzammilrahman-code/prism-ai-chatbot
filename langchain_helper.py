# langchain_helper.py imports
import os
from dotenv import load_dotenv
import shutil

from langchain_community.vectorstores import FAISS          # <---  Vector Stores moved to community
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader  # <---  Loaders moved to community
from langchain_community.embeddings import HuggingFaceEmbeddings # <---  Embeddings moved to community
from langchain_core.prompts import PromptTemplate           # <---  Prompts moved to core
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
import csv
# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# but rely on os.environ.get for security.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
   print("CRITICAL ERROR: GROQ_API_KEY is missing from .env!")

VECTODB_FILE_PATH = "faiss_index"
CSV_FILE_PATH = 'prism_ai.csv' # Prism Q&A data is here

# Initialize LLM and Embeddings
llm = ChatGroq(
    model_name="llama-3.1-8b-instant", # Using a standard Groq model name for reliability
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

def create_vector_db():
    loader = CSVLoader(
        file_path=CSV_FILE_PATH,
        encoding="utf8"
    )
    data = loader.load()

    # Transform documents
    for doc in data:
        row = doc.page_content.split("\n")
        q = row[0].replace("prompt: ", "")
        a = row[1].replace("response: ", "")
        doc.page_content = f"Question: {q}\nAnswer: {a}"

    if os.path.exists(VECTODB_FILE_PATH):
        shutil.rmtree(VECTODB_FILE_PATH)

    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(VECTODB_FILE_PATH)


def get_qa_chain():
    """Load vector database and create RetrievalQA chain."""
    
    # --- Load Vector Database ---
    try:
        # NOTE: FAISS requires the embeddings object used for creation to be passed here
        vectordb = FAISS.load_local(VECTODB_FILE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        # Return None if the file doesn't exist (i.e., KB hasn't been created yet)
        return None
    
    # --- Configure Retriever ---
    # k=1 means the retriever fetches the single most relevant document
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    # --- Define Prompt Template ---
    prompt_template = """
    You are the official Prism AI assistant.
    
    RULES (VERY IMPORTANT):
    - You MUST answer using ONLY the provided context.
    - DO NOT guess.
    - DO NOT add extra information.
    - If the answer is NOT clearly present, say: "As a chatbot, I don't know much about that. Kindly contact us via Email."
    
    CONTEXT: {context}

    USER QUESTION: {question}

    FINAL ANSWER (from context only):
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # --- Create Retrieval Chain ---
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' chains all documents into the prompt
        retriever=retriever,
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain

if __name__ == '__main__':
    # This block allows you to test creation outside of Streamlit
    # create_vector_db() 
    pass