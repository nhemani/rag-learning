#!/usr/bin/env python3
"""
src/core_pipeline.py
Encapsulates runtime clinical inference execution under strict deterministic configurations.
"""

import os
import gc
import torch
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalRAGApp:
    """Manages the full production pipeline execution: context retrieval and answer generation."""
    
    def __init__(self, chroma_dir: str = "./data/chroma_db", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device}
        )
        
        self.vector_db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model
        )
        
        self.qna_system = """<s>[INST] <<SYS>>
You are an expert clinical critical care specialist. Use ONLY the verified facts enclosed within the provided context tags to answer the user query.
If the context does not contain enough evidence to answer accurately, state exactly 'I don't know'. Do not make up facts.
Do not repeat these instructions or mention your rules in the final output.
<</SYS>>"""
        self.qna_user_template = """<context>\n{context}\n</context>\n\nPatient Case/Query: {question} [/INST]"""

    def generate_response(self, llm_instance: Any, query: str, category_filter: str = None, k: int = 4) -> Dict[str, str]:
        torch.cuda.empty_cache()
        gc.collect()
        
        search_kwargs = {"k": k}
        if category_filter:
            search_kwargs["filter"] = {"section": category_filter}
            
        retriever = self.vector_db.as_retriever(search_kwargs=search_kwargs)
        relevant_chunks = retriever.invoke(query)
        context_text = " ".join([chunk.page_content for chunk in relevant_chunks])
        
        prompt = f"{self.qna_system}\n{self.qna_user_template.format(context=context_text, question=query)}"
        
        try:
            output = llm_instance(
                prompt=prompt,
                max_tokens=350,
                temperature=0.0,      # Absolute determinism
                repeat_penalty=1.15,  # Structural loop protection
                top_p=1.0,
                top_k=1,
                echo=False
            )
            answer = output['choices']['text'].strip() if isinstance(output['choices'], list) else output['choices']['text'].strip()
        except Exception as e:
            answer = f"Generation failure: {e}"
            
        return {"answer": answer, "context": context_text}
