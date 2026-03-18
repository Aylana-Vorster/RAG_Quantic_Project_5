import time
import json
import numpy as np
import pandas as pd
import os
from rag_engine import RAGEngine
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuration for LLM-as-a-judge
EVAL_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 20 Evaluation Queries with Gold Answers (Expectations)
TEST_DATA = [
    {
        "query": "What is the definition of Done in Sandbox Agile?",
        "gold_answer": "Code is peer-reviewed, unit tests are written and passing, the feature is tested by QA, and the Product Owner has accepted the story.",
        "topic": "Agile"
    },
    {
        "query": "How many days of annual leave are full-time employees entitled to per year in South Africa?",
        "gold_answer": "21 consecutive days of paid annual leave per year (1.75 days per month).",
        "topic": "Leave"
    },
    {
        "query": "What is the maximum number of annual leave days an employee can accumulate?",
        "gold_answer": "A maximum of 30 days of annual leave.",
        "topic": "Leave"
    },
    {
        "query": "When is a medical certificate required for sick leave?",
        "gold_answer": "If absent for more than two consecutive days, or on more than two occasions during an eight-week period.",
        "topic": "Leave"
    },
    {
        "query": "What is the company's dress code policy for regular office days?",
        "gold_answer": "The dress code is 'smart casual', which means professional yet relaxed, neat and put-together.",
        "topic": "Dress Code"
    },
    {
        "query": "What should an employee wear when meeting with clients?",
        "gold_answer": "Business professional attire (e.g., suit, blazer, dress shirt, dress shoes).",
        "topic": "Dress Code"
    },
    {
        "query": "Are flip-flops allowed in the office?",
        "gold_answer": "No, beachwear like flip-flops is specifically listed as unacceptable attire.",
        "topic": "Dress Code"
    },
    {
        "query": "How many days of family responsibility leave are employees entitled to?",
        "gold_answer": "3 days of paid family responsibility leave per year, for employees who have worked more than four months.",
        "topic": "Leave"
    },
    {
        "query": "Does Sandbox provide equipment for remote work?",
        "gold_answer": "Yes, Sandbox provides necessary equipment including a laptop and other peripherals.",
        "topic": "Remote Work"
    },
    {
        "query": "Who is responsible for the cost of internet service for remote employees?",
        "gold_answer": "Employees are responsible for their own internet service and costs, though a stipend may be provided as outlined in the Expense Policy.",
        "topic": "Remote Work"
    },
    {
        "query": "Within how many days should an expense report be submitted?",
        "gold_answer": "Expense reports should be submitted within 30 days of incurring the expense.",
        "topic": "Expenses"
    },
    {
        "query": "Is childcare a reimbursable expense?",
        "gold_answer": "No, childcare is specifically listed as a non-reimbursable expense.",
        "topic": "Expenses"
    },
    {
        "query": "What are the core Scrum events mentioned in the Agile guide?",
        "gold_answer": "The Sprint, Sprint Planning, Daily Scrum (Stand-up), Sprint Review, and Sprint Retrospective.",
        "topic": "Agile"
    },
    {
        "query": "Who is the CEO of Sandbox?",
        "gold_answer": "Stacy Fakename.",
        "topic": "General"
    },
    {
        "query": "What is the 'Home Office Stipend' mentioned in the Expense Policy?",
        "gold_answer": "A monthly stipend provided to remote employees to cover a portion of home office expenses like internet and electricity.",
        "topic": "Expenses"
    },
    {
        "query": "What are the three questions for the Daily Scrum?",
        "gold_answer": "1. What did I do yesterday? 2. What will I do today? 3. Are there any impediments in my way?",
        "topic": "Agile"
    },
    {
        "query": "What is the 'Scrum Master' responsible for?",
        "gold_answer": "The Scrum Master is a servant-leader responsible for helping everyone understand Scrum and removing impediments to the team's progress.",
        "topic": "Agile"
    },
    {
        "query": "Can an employee be reimbursed for traffic violations?",
        "gold_answer": "No, traffic violations or parking tickets are non-reimbursable.",
        "topic": "Expenses"
    },
    {
        "query": "What is the leave year period at Sandbox?",
        "gold_answer": "The leave year runs from January 1st to December 31st.",
        "topic": "Leave"
    },
    {
        "query": "What is the policy on company space travel and Mars colonies?",
        "gold_answer": "Refusal: The documents do not mention space travel or Mars colonies.",
        "topic": "Out-of-Scope"
    }
]

class Evaluator:
    def __init__(self):
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model_name=EVAL_MODEL, temperature=0)
        
    def judge_groundedness(self, query, answer, context_snippets):
        prompt = ChatPromptTemplate.from_template("""
        You are a fact-checker. Evaluate if the Answer is factually supported by the Context.
        
        Context:
        {context}
        
        Question: {query}
        Answer: {answer}
        
        TASK:
        1. Analyze if every fact in the Answer is present in the Context.
        2. Ignore square brackets and citations.
        3. Provide a brief Rationale for your decision.
        4. End your response with 'VERDICT: YES' or 'VERDICT: NO'.
        """)
        
        context_str = "\n---\n".join([s['content'] for s in context_snippets])
        chain = prompt | self.llm
        response = chain.invoke({"context": context_str, "query": query, "answer": answer})
        
        # Extract verdict from the response
        verdict = "YES" if "VERDICT: YES" in response.content.upper() else "NO"
        return verdict == "YES"

    def judge_citation_accuracy(self, answer, context_snippets):
        # Simplistic check: Does the answer mention at least one of the source filenames from the snippets?
        sources = [s['source'].lower() for s in context_snippets]
        answer_lower = answer.lower()
        for source in sources:
            if source in answer_lower:
                return True
        return False

def run_evaluation():
    try:
        engine = RAGEngine()
        evaluator = Evaluator()
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    results = []
    print(f"Starting detailed evaluation of {len(TEST_DATA)} queries...")

    for item in TEST_DATA:
        query = item["query"]
        print(f"Testing Topic [{item['topic']}]: {query}")
        
        res = engine.query(query)
        
        is_grounded = evaluator.judge_groundedness(query, res["answer"], res["snippets"])
        has_correct_citation = evaluator.judge_citation_accuracy(res["answer"], res["snippets"])
        
        # Partial Match check (simple keyword overlap)
        gold = item["gold_answer"].lower()
        actual = res["answer"].lower()
        # Heuristic: if 30% of words in gold are in actual, call it a partial match
        gold_words = set(gold.replace(',', '').replace('.', '').split())
        match_count = sum(1 for word in gold_words if word in actual)
        partial_match = (match_count / len(gold_words)) > 0.3 if gold_words else True
        
        results.append({
            "topic": item["topic"],
            "query": query,
            "answer": res["answer"],
            "gold_answer": item["gold_answer"],
            "latency": res["latency_ms"],
            "groundedness": is_grounded,
            "citation_accuracy": has_correct_citation,
            "partial_match": partial_match
        })

    df = pd.DataFrame(results)
    
    metrics = {
        "count": len(df),
        "p50_latency_ms": round(np.percentile(df["latency"], 50), 2),
        "p95_latency_ms": round(np.percentile(df["latency"], 95), 2),
        "avg_latency_ms": round(df["latency"].mean(), 2),
        "groundedness_score": round(df["groundedness"].mean() * 100, 2),
        "citation_accuracy_score": round(df["citation_accuracy"].mean() * 100, 2),
        "partial_match_score": round(df["partial_match"].mean() * 100, 2)
    }

    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(json.dumps(metrics, indent=2))
    
    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
