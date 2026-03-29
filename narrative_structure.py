import pandas as pd
from bertopic import BERTopic
import os

# 1. Load your local CSV (No internet needed!)
print(" Loading your 56k articles...")
df = pd.read_csv("filtered_grievance_data_en.csv")

# 2. Data Cleaning for the Model
# Ensure we have dates and text
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# 3. Subsampling for the Demo
# To keep this fast (under 10 mins), let's use 4,000 articles
df_sample = df.sample(n=min(4000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

print(f" Training BERTopic on {len(docs)} articles...")

# 4. Run BERTopic
# This uses 'all-MiniLM-L6-v2' by default - small, fast, and accurate
topic_model = BERTopic(
    language="english", 
    min_topic_size=20, # Smaller size for smaller sample
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)

# 5. Get the Narrative Summary
topic_info = topic_model.get_topic_info()
print("\n---  TOP 5 NARRATIVES IDENTIFIED ---")
print(topic_info[['Topic', 'Count', 'Name']].head(6))

# 6. Save Narrative Over Time (For your VAR model)
print("\n Mapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv("narrative_pulse_data.csv")

print("\n SUCCESS! 'narrative_pulse_data.csv' is ready for your VAR model.")