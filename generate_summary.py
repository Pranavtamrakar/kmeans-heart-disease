import ollama
import pandas as pd
import pickle

# ─────────────────────────────────────────────
# STEP 1 — Load cluster analysis data
# ─────────────────────────────────────────────
cluster_analysis = pd.read_excel('cluster_analysis.xlsx')

print("✅ Data loaded. Columns found:", cluster_analysis.columns.tolist())

# ─────────────────────────────────────────────
# STEP 2 — Generate LLM summary per cluster
# ─────────────────────────────────────────────
cluster_summaries = {}

for cluster_id, group in cluster_analysis.groupby('Cluster'):

 stats = group.mean(numeric_only=True).round(2).to_string()
 print(f"\n⏳ Generating summary for Cluster {cluster_id}...")

 response = ollama.chat(
 model='llama3',
 messages=[{
 'role': 'user',
 'content': (
 f"You are a medical data analyst. "
 f"Summarize the following average statistics for Cluster {cluster_id} "
 f"in plain English using bullet points. "
 f"Focus on what makes this cluster medically distinct:\n\n{stats}"
 )
 }]
 )

 cluster_summaries[cluster_id] = response['message']['content']
 print(f"✅ Cluster {cluster_id} done.")

# ─────────────────────────────────────────────
# STEP 3 — Save to pickle
# ─────────────────────────────────────────────
with open('cluster_summaries.pkl', 'wb') as f:
 pickle.dump(cluster_summaries, f)

print("\n🎉 cluster_summaries.pkl saved successfully!")