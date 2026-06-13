# Assuming your local llama model variable is instantiated and named 'llm'
# Initialize the production pipeline app asset
app = MedicalRAGApp(chroma_dir="./chroma_db")

# Setup a test payload record
test_question = "What is the protocol for managing sepsis in a critical care unit?"
target_domain = "Critical Care Medicine"

# Execute single monolithic run call
pipeline_result = app.run_pipeline(llm_instance=llm, query=test_question, category_filter=target_domain, k=4)

# Render results in a scannable dashboard printout
print("\n" + "="*80)
print("              PRODUCTION SYSTEM OUTPUT RUNTIME METRICS LOGS            ")
print("="*80)
print(f"📍 USER QUESTION : {pipeline_result['query']}")
print(f"📍 TARGET FILTER  : {pipeline_result['category']}")
print("-" * 80)
print(f"📊 GENERATED RAG ANSWER:\n{pipeline_result['rag_answer']}\n")
print("-" * 80)
print(f"🧪 JUDGE: GROUNDEDNESS PROFILE:\n{pipeline_result['groundedness_audit']}\n")
print(f"🧪 JUDGE: RELEVANCE PROFILE:\n{pipeline_result['relevance_audit']}\n")
print("="*80)
