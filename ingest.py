import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
CORPUS_DIR = "corpus"
PDF_FILE = "b516ccaf2905b8f2e096654299cd902a.pdf"
DB_DIR = "db"

def ingest_documents():
    # 1. Load Markdown files
    print("Loading Markdown files...")
    md_loader = DirectoryLoader(CORPUS_DIR, glob="**/*.md", loader_cls=TextLoader)
    md_documents = md_loader.load()
    
    # 2. Split Markdown by headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    all_chunks = []
    for doc in md_documents:
        chunks = md_splitter.split_text(doc.page_content)
        # Add source metadata
        source_name = os.path.basename(doc.metadata['source'])
        for chunk in chunks:
            chunk.metadata['source'] = source_name
            all_chunks.append(chunk)

    # 3. Load PDF file
    if os.path.exists(PDF_FILE):
        print(f"Loading PDF file: {PDF_FILE}...")
        pdf_loader = PyPDFLoader(PDF_FILE)
        pdf_documents = pdf_loader.load()
        
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        pdf_chunks = pdf_splitter.split_documents(pdf_documents)
        
        # Add source metadata for PDF chunks
        for chunk in pdf_chunks:
            chunk.metadata['source'] = PDF_FILE
            all_chunks.append(chunk)
    else:
        print(f"Warning: PDF file {PDF_FILE} not found.")

    # 4. Create Embeddings
    print("Creating embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Store in Chroma
    print(f"Storing {len(all_chunks)} chunks in {DB_DIR}...")
    if os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)
        
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
