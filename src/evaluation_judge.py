#!/usr/bin/env python3
"""
src/evaluation_judge.py
Automated evaluation module scoring pipeline outputs on Groundedness and Relevance.
"""

from typing import Dict, Any

class ClinicalRAGJudge:
    """Enforces zero-VRAM footprint automated quality grading using internal models as self-auditors."""
    
    def __init__(self):
        self.payload_template = "\nMedical Question: {question}\nRetrieved Context: {context}\nGenerated Answer: {answer}\n"
        
        self.groundedness_prompt = """<s>[INST] <<SYS>>
You are an expert clinical auditor evaluating an 'Answer' against the provided 'Context'.
CRITICAL RULES:
1. If the 'Answer' states "I don't know" or refuses to answer because info is missing, you MUST award a score of 5. Refusing to guess is perfectly grounded.
2. If the 'Answer' fabricates any facts not explicitly inside the context, award a score of 1.
Rate on a strict scale from 1 to 5. Format output EXACTLY as: 'Groundedness Score: [Score]' followed by a 1-sentence rationale.
<</SYS>>"""

        self.relevance_prompt = """<s>[INST] <<SYS>>
You are an expert clinical auditor evaluating if the 'Context' and 'Answer' address the specific medical intent of the 'Question'.
Rate on a scale from 1 to 5 (1=completely irrelevant, 5=perfectly targeted).
Format output EXACTLY as: 'Relevance Score: [Score]' followed by a 1-sentence rationale.
<</SYS>>"""

    def audit_response(self, llm_instance: Any, query: str, context: str, answer: str) -> Dict[str, str]:
        formatted_payload = self.payload_template.format(question=query, context=context, answer=answer)
        
        g_prompt = f"{self.groundedness_prompt}\n'user': {formatted_payload} [/INST]"
        r_prompt = f"{self.relevance_prompt}\n'user': {formatted_payload} [/INST]"
        
        try:
            g_output = llm_instance(prompt=g_prompt, max_tokens=150, temperature=0.0, top_p=1.0, top_k=1)
            g_report = g_output['choices']['text'].strip()
            
            r_output = llm_instance(prompt=r_prompt, max_tokens=150, temperature=0.0, top_p=1.0, top_k=1)
            r_report = r_output['choices']['text'].strip()
        except Exception as e:
            g_report = f"Audit failed: {e}"
            r_report = f"Audit failed: {e}"
            
        return {"groundedness": g_report, "relevance": r_report}
