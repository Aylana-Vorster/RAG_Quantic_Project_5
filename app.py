import os
from flask import Flask, request, jsonify, render_template
from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize RAG Engine lazily
engine = None

def get_engine():
    global engine
    if engine is None:
        try:
            engine = RAGEngine()
        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            return None
    return engine

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
        
    rag_engine = get_engine()
    if rag_engine is None:
        return jsonify({"error": "RAG Engine not initialized. Run ingest.py first or check API key."}), 500
        
    try:
        result = rag_engine.query(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    rag_engine = get_engine()
    status = "ok" if rag_engine else "error (check logs/ingestion)"
    db_exists = os.path.exists("db")
    
    return jsonify({
        "status": status,
        "database_exists": db_exists,
        "engine_ready": rag_engine is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
