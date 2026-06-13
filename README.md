# Clinical RAG Pipeline: Advanced Evaluation & Hyperparameter Framework

An enterprise-grade, fully auditable Retrieval-Augmented Generation (RAG) framework optimized for resource-constrained environments (Google Colab T4 GPU). This architecture converts a volatile, probabilistic Local Large Language Model (Llama-2-13B) into a safe, factual clinical lookup utility indexing *The Merck Manual of Diagnosis & Therapy (19th Edition)*.

---

## 🚀 Key Project Milestones
- **Eliminated Hallucinations:** Eradicated dangerous baseline suggestions (e.g., prescribing intravenous hospital narcotics on wilderness trails or performing tumor resections for trauma patients).
- **Hardened Refusal Logic:** Solved prompt leakage and semantic loop errors, forcing the model to output a verified `I don't know` instead of fabricating non-medical info.
- **Automated Self-Auditing:** Designed an internal, zero-VRAM cost **LLM-as-a-Judge** scoring array to evaluate pipeline extraction health in real-time.

---

## 📊 Evaluation & Performance Matrix

| Query ID & Category | Clinical Target / Question | Groundedness | Relevance | Architectural Mitigation & Structural Win |
| :--- | :--- | :---: | :---: | :--- |
| **Query 1: Critical Care** | ICU Sepsis Triage Protocol & Monitoring | **5 / 5** | **5 / 5** | Fixed prompt leakage. Grouped vital indicators into a cohesive clinical workflow. |
| **Query 2: Gastroenterology**| Diagnostic Criteria for Appendicitis | **5 / 5** | **5 / 5** | Removed ungrounded claims that a normal inflamed appendix is a superficial palpable mass. |
| **Query 3: Dermatology** | Sudden Patchy Hair Loss Interventions | **5 / 5** | **5 / 5** | **K=4 Optimization** resolved token truncation bugs across broad multi-page chapters. |
| **Query 4: Neurology** | Traumatic Brain Injury (TBI) Management | **5 / 5** | **5 / 5** | Segregated medical domains, stopping trauma from blending into oncology sub-specialties. |
| **Query 5: Wilderness** | On-Trail Leg Fracture Field Splinting | **5 / 5** | **5 / 5** | Blocked high-risk hallucinations regarding wilderness field tourniquets and controlled narcotics. |
| **Negative Edge-Case** | Java List/Map Classes vs. Coding SOLID Rules | **5 / 5** | **1 / 5** | **Safe Refusal Verified:** Calibrated judge prompt accurately rewards honest negative refusals without penalty. |

---

## 📁 Repository Directory Structure

```text
clinical-rag-pipeline/
│
├── config/
│   └── parameters.yaml          # Pipeline hyperparameter configurations
├── data/
│   ├── raw/
│   │   └── merck_manual_19th.pdf # Source document volume file
│   └── chroma_db/               # Persistent Vector store directory (.gitignored)
├── src/
│   ├── __init__.py              # Package initialization hook
│   ├── database_builder.py      # Independent ingestion & data parsing module
│   ├── core_pipeline.py         # Primary deterministic inference runtime engine
│   └── evaluation_judge.py      # Self-auditing judge prompts and routines
├── notebooks/
│   ├── 01_data_ingestion.ipynb  
│   └── 02_pipeline_testing.ipynb
├── reports/
│   └── clinical_rag_evaluation_report.csv  # Auto-generated batch log workbook
├── .gitignore                   # Safe cache and large vector block ignore maps
├── requirements.txt             # Framework installation tracking manifests
└── README.md                    # Core project presentation dashboard
```

---

## 🛠️ Production Run Configuration

To maintain perfect validation profiles, ensure your inference variables mirror these calibrated constraints:

```python
production_hyperparameters = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retrieval_depth_k": 4,
    "generation_temperature": 0.0,   # Eliminates probabilistic drifting
    "repetition_penalty": 1.15,      # Stop loops during data processing
    "max_tokens_budget": 1024
}
```

---

## 🚀 Quickstart Installation Guide

Clone this repository and spin up your validation loop locally:

```bash
# 1. Clone package files
git clone https://github.com
cd clinical-rag-pipeline

# 2. Extract dependencies
pip install -r requirements.txt

# 3. Compile your vector space database chunk indexes
python src/database_builder.py
```
