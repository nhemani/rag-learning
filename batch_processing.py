import os
import pandas as pd

# 1. Define the 5 explicit problem-statement queries paired with their metadata domains
problem_statements = [
    {
        "category": "Critical Care Protocols",
        "query": "What is the protocol for managing sepsis in a critical care unit?"
    },
    {
        "category": "Diagnostic Assistance",
        "query": "What are the common symptoms and treatments for pulmonary embolism?"
    },
    {
        "category": "Drug Information",
        "query": "Can you provide the trade names of medications used for treating hypertension?"
    },
    {
        "category": "Treatment Plans",
        "query": "What are the first-line options and alternatives for managing rheumatoid arthritis?"
    },
    {
        "category": "Specialty Knowledge",
        "query": "What are the diagnostic steps for suspected endocrine disorders?"
    }
]

# 2. Initialize an empty list to gather processing payloads
batch_records = []

print("🚀 Starting automated clinical batch-evaluation loop...")
print(f"Total queries queued for processing: {len(problem_statements)}\n" + "="*60)

# 3. Execute the batch loop
for index, item in enumerate(problem_statements, start=1):
    category = item["category"]
    query = item["query"]
    
    print(f"[{index}/5] Processing Category: '{category}'")
    print(f"    Query: '{query}'")
    
    try:
        # Fire monolithic pipeline (uses your pre-instantiated 'app' and 'llm' objects)
        result = app.run_pipeline(
            llm_instance=llm, 
            query=query, 
            category_filter=category, 
            k=4
        )
        
        # Append the structured dictionary output to our master records list
        batch_records.append({
            "Query_ID": f"Q_{index}",
            "Category": result["category"],
            "User_Query": result["query"],
            "Generated_RAG_Answer": result["rag_answer"],
            "Groundedness_Audit_Report": result["groundedness_audit"],
            "Relevance_Audit_Report": result["relevance_audit"]
        })
        print(f"    ✓ Successfully evaluated and audited.")
        
    except Exception as e:
        print(f"    ❌ Pipeline failed for this record: {e}")
        batch_records.append({
            "Query_ID": f"Q_{index}",
            "Category": category,
            "User_Query": query,
            "Generated_RAG_Answer": f"Pipeline Error: {e}",
            "Groundedness_Audit_Report": "N/A",
            "Relevance_Audit_Report": "N/A"
        })

print("="*60 + "\n✓ All evaluations completed. Compiling final metrics table...")

# 4. Convert records list into a structured Pandas DataFrame
df_evaluations = pd.DataFrame(batch_records)

# 5. Export to an external CSV file
output_csv_path = "./clinical_rag_evaluation_report.csv"
df_evaluations.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"🎉 Success! The final validation matrix has been saved to: '{output_csv_path}'")
