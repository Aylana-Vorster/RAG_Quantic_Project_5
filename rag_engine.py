import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_DIR = "db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

class RAGEngine:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        # Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load Vector Store
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(f"Vector store directory {DB_DIR} not found. Run ingest.py first.")
            
        self.vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0
        )
        
        # Custom Prompt Template
        prompt_template = """You are a helpful assistant for Sandbox company. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, politely refuse to answer and state that you can only provide information from company policy documents.
Always cite the source document names/IDs for your answer.

Context:
{context}

Question: {question}

Helpful Answer (include citations at the end or in-line):"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Retrieval QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question):
        start_time = time.time()
        
        response = self.qa_chain.invoke({"query": question})
        
        latency = (time.time() - start_time) * 1000  # ms
        
        answer = response["result"]
        source_docs = response["source_documents"]
        
        citations = []
        snippets = []
        for doc in source_docs:
            source = doc.metadata.get("source", "Unknown")
            citations.append(source)
            snippets.append({
                "source": source,
                "content": doc.page_content[:200] + "..."
            })
            
        # Deduplicate citations
        unique_citations = list(set(citations))
        
        return {
            "answer": answer,
            "citations": unique_citations,
            "snippets": snippets,
            "latency_ms": round(latency, 2)
        }

if __name__ == "__main__":
    # Test if script is run directly
    try:
        engine = RAGEngine()
        result = engine.query("What is the definition of Done in Agile?")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
    except Exception as e:
        print(f"Error: {e}")
