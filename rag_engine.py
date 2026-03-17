import os
import time
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_DIR = "db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
OR_MODEL = "meta-llama/llama-3.3-70b-instruct" # Common free/cheap OpenRouter model

class RAGEngine:
    def __init__(self):
        # Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load Vector Store
        if not os.path.exists(DB_DIR):
            raise FileNotFoundError(f"Vector store directory {DB_DIR} not found. Run ingest.py first.")
            
        self.vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=self.embeddings
        )
        
        # Initialize Primary (Groq)
        self.primary_llm = None
        if GROQ_API_KEY:
            self.primary_llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0
            )
            
        # Initialize Fallback (OpenRouter)
        self.fallback_llm = None
        if OPENROUTER_API_KEY:
            self.fallback_llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                model=OR_MODEL,
                temperature=0
            )

        if not self.primary_llm and not self.fallback_llm:
            raise ValueError("No LLM API keys found in .env. Please provide GROQ_API_KEY or OPENROUTER_API_KEY.")

        # Default to primary for initial chain
        self._build_chain(self.primary_llm or self.fallback_llm)

    def _build_chain(self, llm):
        prompt_template = """You are a helpful company policy expert for Sandbox. 
Your goal is to provide accurate answers based ONLY on the provided context.

INSTRUCTIONS:
1. Answer the question using the provided context snippets.
2. For EVERY fact or sentence, you MUST include the source filename in square brackets, e.g., [leave_policy.md].
3. If the answer is not in the context, say "I am sorry, but the company documents do not contain information on this topic." 
4. Do NOT use any external knowledge.
5. Be concise and professional.

Context:
{context}

Question: {question}

Helpful Answer with [Filenames]:"""
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def query(self, question):
        start_time = time.time()
        
        try:
            # Try Primary
            response = self.qa_chain.invoke({"query": question})
            provider = "Groq"
        except Exception as e:
            # Check if it's a rate limit or if we should fallback
            if self.fallback_llm:
                print(f"Primary LLM failed ({e}). Switching to Fallback...")
                self._build_chain(self.fallback_llm)
                response = self.qa_chain.invoke({"query": question})
                provider = "OpenRouter"
            else:
                raise e
        
        latency = (time.time() - start_time) * 1000
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
            
        return {
            "answer": answer,
            "citations": list(set(citations)),
            "snippets": snippets,
            "latency_ms": round(latency, 2),
            "provider": provider
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
