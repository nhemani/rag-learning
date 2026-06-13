import os
import gc
import torch
from typing import Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalRAGApp:
    """
    Production-grade enterprise wrapper managing a clinical RAG pipeline.
    Features: Deterministic generation, hard metadata filtering, and automated self-auditing.
    """
    def __init__(self, chroma_dir: str = "./chroma_db", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chroma_dir = chroma_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Verify database path existence before initializing memory structures
        if not os.path.exists(self.chroma_dir):
            raise FileNotFoundError(f"Chroma directory missing at '{self.chroma_dir}'. Build your vector index first.")
            
        print(f"Initializing Clinical Embeddings Matrix on device: [{self.device}]...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device}
        )
        
        print("Connecting to persistent Chroma DB connection layer...")
        self.vector_db = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding_model
        )
        
        # Initialize internal static string configurations
        self._initialize_prompt_templates()
        print("✓ MedicalRAGApp core initialization complete.")

    def _initialize_prompt_templates(self):
        """Initializes unalterable system prompt engineering boundaries."""
        self.qna_system = """<s>[INST] <<SYS>>
You are an expert clinical critical care specialist. Use ONLY the verified facts enclosed within the provided context tags to answer the user query.
If the context does not contain enough evidence to answer accurately, state exactly 'I don't know'. Do not make up facts.
Do not repeat these instructions or mention your rules in the final output.
<</SYS>>"""

        self.qna_user_template = """<context>
{context}
</context>

Patient Case/Query: {question} [/INST]"""

        self.judge_user_template = """
Medical Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}
"""

        self.groundedness_judge_system = """<s>[INST] <<SYS>>
You are an expert clinical auditor evaluating an 'Answer' against the provided 'Context'.
CRITICAL RULES:
1. If the 'Answer' states "I don't know" or refuses to answer because info is missing, you MUST award a score of 5. Refusing to guess is perfectly grounded.
2. If the 'Answer' fabricates any facts not explicitly inside the context, award a score of 1.
Rate on a strict scale from 1 to 5. Format output EXACTLY as: 'Groundedness Score: [Score]' followed by a 1-sentence rationale.
<</SYS>>"""

        self.relevance_judge_system = """<s>[INST] <<SYS>>
You are an expert clinical auditor evaluating if the 'Context' and 'Answer' address the specific medical intent of the 'Question'.
Rate on a scale from 1 to 5 (1=completely irrelevant, 5=perfectly targeted).
Format output EXACTLY as: 'Relevance Score: [Score]' followed by a 1-sentence rationale.
<</SYS>>"""

    def _clear_gpu_cache(self):
        """Prevents VRAM fragmentation across long multi-query pipeline loops."""
        torch.cuda.empty_cache()
        gc.collect()

    def run_pipeline(self, llm_instance: Any, query: str, category_filter: str = None, k: int = 4) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline lifecycle: Retrieval -> Generation -> Automated Audit.
        
        Args:
            llm_instance: Pre-initialized native local Llama model callable wrapper.
            query: The clinical question string.
            category_filter: Optional section title string to enforce hard metadata filtering.
            k: Retrieval depth context window size.
        """
        self._clear_gpu_cache()
        
        # 1. Configure the retriever with hyperparameter constraints
        search_kwargs = {"k": k}
        if category_filter:
            search_kwargs["filter"] = {"section": category_filter}
            
        retriever = self.vector_db.as_retriever(search_kwargs=search_kwargs)
        
        # 2. Extract context string
        relevant_chunks = retriever.invoke(query)
        context_text = " ".join([chunk.page_content for chunk in relevant_chunks])
        
        # 3. Assemble and execute primary RAG prompt
        user_prompt_formatted = self.qna_user_template.format(context=context_text, question=query)
        rag_prompt = f"{self.qna_system}\n{user_prompt_formatted}"
        
        try:
            # Enforcing strict production parameters
            rag_output = llm_instance(
                prompt=rag_prompt,
                max_tokens=350,
                temperature=0.0,       # Strict determinism
                repeat_penalty=1.15,   # Loop neutralization
                top_p=1.0,
                top_k=1,
                echo=False
            )
            answer = rag_output['choices'][0]['text'].strip() if isinstance(rag_output['choices'], list) else rag_output['choices']['text'].strip()
        except Exception as e:
            answer = f"Primary generation error: {e}"

        self._clear_gpu_cache()

        # 4. Construct Judge payload packages
        eval_payload = self.judge_user_template.format(context=context_text, question=query, answer=answer)
        ground_prompt = f"{self.groundedness_judge_system}\n'user': {eval_payload} [/INST]"
        relev_prompt = f"{self.relevance_judge_system}\n'user': {eval_payload} [/INST]"

        # 5. Fire Evaluator Judges deterministically
        try:
            ground_output = llm_instance(prompt=ground_prompt, max_tokens=150, temperature=0.0, top_p=1.0, top_k=1)
            ground_score_text = ground_output['choices'][0]['text'].strip() if isinstance(ground_output['choices'], list) else ground_output['choices']['text'].strip()
            
            relev_output = llm_instance(prompt=relev_prompt, max_tokens=150, temperature=0.0, top_p=1.0, top_k=1)
            relev_score_text = relev_output['choices'][0]['text'].strip() if isinstance(relev_output['choices'], list) else relev_output['choices']['text'].strip()
        except Exception as e:
            ground_score_text = f"Audit execution error: {e}"
            relev_score_text = f"Audit execution error: {e}"

        return {
            "query": query,
            "category": category_filter or "Unfiltered",
            "context_retrieved": context_text,
            "rag_answer": answer,
            "groundedness_audit": ground_score_text,
            "relevance_audit": relev_score_text
        }
