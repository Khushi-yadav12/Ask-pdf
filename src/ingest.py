import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into ChromaDB.")
    parser.add_argument("--clear", action="store_true", help="Clear the database before ingestion")
    args = parser.parse_args()
    
    if args.clear:
        print("Clearing database...")
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)

    # 1. Load PDFs
    if not os.path.exists(DATA_PATH):
        print(f"Directory {DATA_PATH} not found. Creating it.")
        os.makedirs(DATA_PATH)
        
    print(f"Loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {DATA_PATH}. Please add some PDFs and try again.")
        return
        
    print(f"Loaded {len(documents)} document pages.")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Create embeddings and save to Chroma
    print("Creating embeddings and saving to ChromaDB...")
    # NOTE: You can use other Gemini embedding models if preferred
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create or update the vector store
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Ingestion complete. Vector store saved to {CHROMA_PATH}/")

if __name__ == "__main__":
    main()
