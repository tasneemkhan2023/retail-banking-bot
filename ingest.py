import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the API Key from .env file
load_dotenv()

def create_vector_db():
    # 1. Check if 'data' folder exists and has PDFs
    if not os.path.exists('./data') or not os.listdir('./data'):
        print("❌ Error: 'data' folder is missing or empty. Please add your bank PDFs there.")
        return

    print("--- 📂 Step 1: Loading PDFs from 'data' folder ---")
    # Using DirectoryLoader to grab all PDFs at once
    loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Successfully loaded {len(documents)} pages from your documents.")

    print("--- ✂️ Step 2: Chunking Documents (NLP Logic) ---")
    # Splitting text into chunks of 800 characters with a small overlap
    # This helps the AI maintain context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks for the database.")

    print("--- 🧠 Step 3: Generating Embeddings (Deep Learning) ---")
    # This runs LOCALLY on your computer using Hugging Face
    # It turns text into mathematical vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("--- 💾 Step 4: Saving to ChromaDB ---")
    # This creates the 'chroma_db' folder and stores the vectors
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("\n✅ Success! Your Retail Banking Knowledge Base is ready.")
    print("A folder named 'chroma_db' has been created in your project directory.")

if __name__ == "__main__":
    create_vector_db()