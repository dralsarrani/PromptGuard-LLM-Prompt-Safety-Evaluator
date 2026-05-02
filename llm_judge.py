#pip install openai

import os
import json
from openai import OpenAI
from rag_pipeline import load_vector_store, retrieve_similar
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_PROMPT = """
You are a prompt safety evaluator for LLM systems.
 
Your job is to evaluate whether a given user prompt is SAFE or UNSAFE.
 
You will be given:
1. The prompt to evaluate
2. A list of similar prompts from a safety dataset, each labeled SAFE or UNSAFE
 
Use the similar prompts as context clues, but make your own judgment.
 
You MUST respond in valid JSON only, with exactly this structure:
{
  "verdict": "SAFE" or "UNSAFE",
  "confidence": a number between 0.0 and 1.0,
  "category": one of ["jailbreak", "harmful_content", "privacy_violation", "misinformation", "social_engineering", "safe"],
  "reasoning": "2-3 sentences explaining your verdict clearly"
}
 
Do not include any text outside the JSON. No preamble, no explanation, just the JSON object.
""".strip()


# BUILD CONTEXT FROM RETRIEVED PROMPTS 
 
def build_context(similar_prompts: list) -> str:
    lines = ["Similar prompts from safety dataset:\n"]
    for i, item in enumerate(similar_prompts, 1):
        lines.append(
            f"{i}. [{item['label']}] (similarity: {item['similarity']}) "
            f"\"{item['prompt'][:120]}\""
        )
    return "\n".join(lines)
 

 
# MAIN JUDGE FUNCTION
 
def evaluate_prompt(user_prompt: str, collection, model) -> dict:
    """
    Full pipeline: retrieve similar prompts → call ai → return structured verdict
    """
    # retrieve similar prompts from vector store
    similar = retrieve_similar(user_prompt, collection, model, top_k=5)
    context = build_context(similar)
 
    # build the user message
    user_message = f"""
    Prompt to evaluate:
    \"{user_prompt}\"
 
    {context}
 
    Evaluate the prompt and return your verdict as JSON.
    """.strip()
 
    # call ai
    response = client.chat.completions.create(
        model      = MODEL,
        max_tokens = 1000,
        messages   = [ {"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": user_message}],
    )
 
    raw = response.choices[0].message.content.strip()

    # parse JSON safely
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if ai adds any extra text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        result = json.loads(raw[start:end])
 
    # Attach retrieved context to result for transparency
    result["retrieved_examples"] = similar
 
    return result
 
 
# PRETTY PRINT
 
def print_verdict(prompt: str, result: dict):
    verdict = result["verdict"]
    confidence = result["confidence"]
    category = result["category"]
    reasoning = result["reasoning"]
 
    color = "\033[91m" if verdict == "UNSAFE" else "\033[92m"
    reset = "\033[0m"
 
    print(f"\n{'─'*60}")
    print(f"Prompt   : {prompt[:100]}")
    print(f"Verdict  : {color}{verdict}{reset}  (confidence: {confidence})")
    print(f"Category : {category}")
    print(f"Reasoning: {reasoning}")
    print(f"\nTop retrieved examples:")
    for ex in result["retrieved_examples"][:3]:
        print(f"  [{ex['label']}] sim={ex['similarity']} | {ex['prompt'][:80]}...")
 
