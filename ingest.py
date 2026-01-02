import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_data():
    documents = []
    
    import glob
    # Load 2024 PDF
    pdf_files = glob.glob("data/2024*.pdf")
    if pdf_files:
        pdf_path = pdf_files[0]
        print(f"Loading {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    else:
        print(f"Warning: No 2024 PDF found in data/.")

    # Load 2025 Text
    txt_path = "data/2025년 개정세법.txt"
    if os.path.exists(txt_path):
        print(f"Loading {txt_path}...")
        loader = TextLoader(txt_path, encoding='utf-8')
        documents.extend(loader.load())
    else:
        print(f"Warning: {txt_path} not found.")

    if not documents:
        print("No documents loaded.")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")

    # Create Embeddings and Vector Store
    embeddings = OpenAIEmbeddings()
    persist_directory = "chroma_db"
    
    print("Creating vector store...")
    
    # Batch processing to avoid rate limits
    batch_size = 100
    import time

    total_chunks = len(splits)
    vectordb = None
    
    for i in range(0, total_chunks, batch_size):
        batch = splits[i:i+batch_size]
        print(f"Processing batch {i}/{total_chunks}...")
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            vectordb.add_documents(batch)
        time.sleep(1) # Sleep to respect rate limits

    print(f"Vector store created in {persist_directory}")

if __name__ == "__main__":
    ingest_data()
