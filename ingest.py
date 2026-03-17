import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configuration
CORPUS_DIR = "corpus"
DB_DIR = "db"

def ingest_documents():
    all_chunks = []
    
    # 1. Load and process Markdown files from corpus/
    print(f"Searching for Markdown files in {CORPUS_DIR}...")
    md_loader = DirectoryLoader(CORPUS_DIR, glob="**/*.md", loader_cls=TextLoader)
    md_documents = md_loader.load()
    
    if md_documents:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        for doc in md_documents:
            chunks = md_splitter.split_text(doc.page_content)
            source_name = os.path.basename(doc.metadata['source'])
            for chunk in chunks:
                # Prepend source to content to help LLM attribution
                chunk.page_content = f"[Source: {source_name}]\n{chunk.page_content}"
                chunk.metadata['source'] = source_name
                all_chunks.append(chunk)
        print(f"Processed {len(md_documents)} Markdown files.")

    # 2. Load and process PDF files from corpus/
    print(f"Searching for PDF files in {CORPUS_DIR}...")
    pdf_loader = DirectoryLoader(CORPUS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()
    
    if pdf_documents:
        pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        pdf_chunks = pdf_splitter.split_documents(pdf_documents)
        
        for chunk in pdf_chunks:
            source_name = os.path.basename(chunk.metadata['source'])
            # Prepend source to content
            chunk.page_content = f"[Source: {source_name}]\n{chunk.page_content}"
            chunk.metadata['source'] = source_name
            all_chunks.append(chunk)
        print(f"Processed {len(pdf_documents)} PDF pages.")

    if not all_chunks:
        print("No documents found in the corpus directory.")
        return

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
