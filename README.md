# 🛡️ PromptGuard — LLM Prompt Safety Evaluator

> Detect unsafe prompts before they reach your LLM powered by RAG retrieval and an AI judge, Used 181K labeled prompts.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/dralsarrani/PromptGuard)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-181k%20Prompts-yellow)](https://huggingface.co/datasets/dralsarrani/Prompt-Aggregation-Dataset-Custom-Dataset)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)

![PromptGuard UI](PromptGuard.gif)

---

## What is PromptGuard?

PromptGuard is a production-ready prompt safety evaluation system. It takes any user prompt and determines whether it is **safe or unsafe** returning a confidence score, a harm category, and a human-readable explanation of the reasoning.

Unlike simple keyword filters or basic classifiers, PromptGuard uses a **RAG + LLM judge pipeline**: it retrieves the most similar prompts from a 181k labeled dataset as context, then uses an LLM to reason about safety using that evidence. This makes it both more accurate and more explainable than traditional approaches.

---

## How It Works

```
User Prompt
     │
     ▼
┌─────────────────────────────┐
│   RAG Retrieval (ChromaDB)  │  ← finds top 5 similar prompts from 181k dataset
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│     LLM Judge (via API)     │  ← evaluates safety using retrieved context
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│     Structured Verdict      │  ← verdict + confidence + category + reasoning
└─────────────────────────────┘
```

**Step 1 — Embedding & Retrieval**
The input prompt is embedded using `sentence-transformers` and compared against 181K labeled prompts stored in a ChromaDB vector store. The top 5 most semantically similar prompts are retrieved along with their labels.

**Step 2 — LLM Judge**
The retrieved examples are passed as context to an LLM judge alongside the original prompt. The judge reasons about whether the prompt is harmful based on pattern similarity and semantic intent not just keywords.

**Step 3 — Structured Output**
The system returns a JSON verdict:
```json
{
  "verdict": "UNSAFE",
  "confidence": 0.97,
  "category": "harmful_content",
  "reasoning": "The prompt explicitly requests instructions for creating illegal substances..."
}
```

---

## Output Example

| Field | Value |
|---|---|
| **Verdict** | 🚨 UNSAFE |
| **Confidence** | 97% |
| **Category** | Harmful Content |
| **Reasoning** | The prompt requests step-by-step synthesis instructions for a controlled substance. Similar prompts in the dataset are consistently labeled unsafe. |

---

## Harm Categories

| Category | Description |
|---|---|
| 🔓 `jailbreak` | Attempts to bypass LLM safety guardrails |
| ☠️ `harmful_content` | Requests for dangerous or illegal instructions |
| 🕵️ `privacy_violation` | Attempts to extract personal or sensitive data |
| 🧪 `misinformation` | Prompts designed to generate false information |
| 🎭 `social_engineering` | Manipulation or phishing-style prompts |
| ✅ `safe` | No harmful intent detected |

---

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| LLM Judge | OpenRouter API |
| UI | Gradio |
| Deployment | HuggingFace Spaces |
| Dataset | 181k custom prompt safety dataset |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/dralsarrani/PromptGuard-LLM-Prompt-Safety-Evaluator
cd PromptGuard-LLM-Prompt-Safety-Evaluator
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your API key**
```bash
export OPENROUTER_API_KEY="your-key-here"
```

**4. Build the vector store (first run only)**
```bash
python rag_pipeline.py
```

**5. Launch the app**
```bash
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Dataset

This project is built on a custom 181K prompt safety dataset collected and labeled from multiple sources, covering jailbreaks, harmful content, social engineering, misinformation, and safe prompts.

👉 [View the dataset on HuggingFace](https://huggingface.co/datasets/dralsarrani/prompt_safety_with_synthetic_labeled) you can also 
👉 [View the dataset docs on GitHub](https://github.com/dralsarrani/Prompt-Aggregation-Dataset---Custom-Dataset)

---

## Live Demo

Try it live, no setup needed:>

👉 **[huggingface.co/spaces/dralsarrani/PromptGuard](https://huggingface.co/spaces/dralsarrani/PromptGuard)**

---

## 🔮 Future Work

- [ ] **Category filtering** let users test for specific threat types only (e.g. jailbreaks only)
- [ ] **PDF export** one-click downloadable safety report
- [ ] **Multilingual support** extend the dataset and evaluation to Arabic and other languages
- [ ] **API endpoint** expose the evaluator as a REST API so developers can integrate it into their own pipelines

---

## Author

**Danah Al-Sarrani**
AI Engineer | LLM & GenAI

[![HuggingFace](https://img.shields.io/badge/🤗-dralsarrani-yellow)](https://huggingface.co/dralsarrani)
[![GitHub](https://img.shields.io/badge/GitHub-dralsarrani-black)](https://github.com/dralsarrani)
