import time
import json
import numpy as np
import pandas as pd
from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()

TEST_QUERIES = [
    "What is the definition of Done in Sandbox Agile?",
    "What is the dress code policy?",
    "How do I report a grievance?",
    "Who is the CEO of Sandbox?",
    "What is the policy on remote work?",
    "Can I use my own laptop for work?",
    "What are the Scrum ceremonies?",
    "How do I refer a candidate for a job?",
    "What is the leave policy for study?",
    "How do I claim expenses?",
    "What should I do if I see a safety hazard?",
    "What is the company policy on space travel?" # Should be refused
]

def evaluate():
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"Error initializing engine: {e}")
        return

    results = []
    print(f"Running evaluation with {len(TEST_QUERIES)} queries...")

    for query in TEST_QUERIES:
        print(f"Testing: {query}")
        res = engine.query(query)
        
        # Simple heuristic for Groundedness: 
        # Check if any citations are present and if the answer isn't a "refusal" for valid policy questions.
        # For the "space travel" query, a refusal is a "correct" grounded response.
        is_refusal = "only provide information from company policy documents" in res["answer"].lower() or \
                     "not mentioned" in res["answer"].lower() or \
                     "sorry" in res["answer"].lower()
        
        is_grounded = True # Default
        if "space travel" in query.lower():
            is_grounded = is_refusal
        elif is_refusal:
            is_grounded = False
            
        has_citation = len(res["citations"]) > 0
        
        results.append({
            "query": query,
            "answer": res["answer"],
            "citations": res["citations"],
            "latency": res["latency_ms"],
            "is_grounded": is_grounded,
            "has_citation": has_citation
        })

    df = pd.DataFrame(results)
    
    metrics = {
        "p50_latency": np.percentile(df["latency"], 50),
        "p95_latency": np.percentile(df["latency"], 95),
        "avg_latency": df["latency"].mean(),
        "groundedness_score": df["is_grounded"].mean() * 100,
        "citation_accuracy": df["has_citation"].mean() * 100
    }

    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))
    
    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    evaluate()
