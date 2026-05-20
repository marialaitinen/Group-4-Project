import pandas as pd
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer

# --- FIXING THE PARALLEL PATH ISSUE ---
# 1. Get the path to the 'scripts' folder
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go UP one level and then into the 'data' folder
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Load from the parallel data folder
print(f" Loading your articles from: {data_dir}")
input_path = os.path.join(data_dir, "tagged_grievance_data_en.csv")
df = pd.read_csv(input_path)

# Data Cleaning for the Model
# Ensure we have dates and text
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling
# Increased to 10,000 for better VAR density
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()
domain = df_sample["url"].tolist()

print(f" Training BERTopic on {len(docs)} articles...")

vectorizer = CountVectorizer(stop_words="english")

# THE SEED TOPICS (Separating the "Outcome" from the "Causes")
seed_topic_list = [
    # 0. THE OUTCOME VARIABLE (What we are trying to predict)
    ["protest", "riot", "riots", "strike", "demonstration"], 
    
    # 1-8. THE CAUSATION VARIABLES (Your 8 categories!)
    ["political", "anti-government", "government", "anti-president", "president", "coalition", "opposition", "resignation", "resigns", "impeachment"], # 1: Government
    ["journalist", "journalists", "freedom", "lawyer", "democracy", "law", "independence", "anti-police", "constitution", "anti-corruption", "corruption", "reform", "anti-segregation", "constitutional", "suffrage", "women", "referendum", "fraud", "civil society"], # 2: Dem-Reform
    ["occupy", "wef", "davos", "u.n.", "us", "intervention", "foreign", "anti-globalization", "g20", "climate", "environment", "environmental", "immigration", "brexit", "migration", "migrant", "refugee", "human rights", "summit", "anti-war"], # 3: Global
    ["anti-blasphemy", "mosque", "quran", "religion", "religious"], # 4: Religious
    ["candidates", "vote", "electoral", "poll", "election"], # 5: Elections
    ["anti-austerity", "austerity", "electricity", "energy", "yellow vests", "gas", "union", "healthcare", "education", "school", "land", "agriculture", "basic needs"], # 6: Basic Needs
    ["ousted", "assassination", "assassinated", "military", "coup", "tenure"], # 7: Coup
    ["deadly", "violent", "civil war", "burning", "violence"] # 8: Violence
]

# Run BERTopic
topic_model = BERTopic(
    language="english", 
    min_topic_size=20, 
    verbose=True,
    vectorizer_model=vectorizer,
    seed_topic_list=seed_topic_list
)

topics, probs = topic_model.fit_transform(docs)
topics = topic_model.topics_

df_sample = df_sample.copy()
df_sample["topic"] = topics 
doc_info = topic_model.get_document_info(docs) 
df_sample["probability"] = doc_info["Probability"]

# Get the Narrative Summary
topic_info = topic_model.get_topic_info()
print("\n---  TOP 5 NARRATIVES IDENTIFIED ---")
print(topic_info[['Topic', 'Count', 'Name']].head(6))

# Save Narrative Over Time (Pointing to data folder)
print("\n Mapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_en.csv"))

print("\n SUCCESS! 'narrative_pulse_data_en.csv' is ready for your VAR model.")

# Print Topic Details
for topic_id in df_sample["topic"].unique():
    if topic_id == -1: 
        continue
    print(f"topic {topic_id}")
    top_articles = (
        df_sample[df_sample["topic"] == topic_id]
        .sort_values("probability", ascending=False)
        .head(5)
    )
                             
    for i, row in top_articles.iterrows():
        print("for n: \n")
        print(row["url"])  
        print(row["maintext"][:300])

# Save the full sample with topic tags (Pointing to data folder)
df_sample.to_csv(os.path.join(data_dir, "classified_articles_en.csv"), index=False)
        
df_sample['date_publish'] = pd.to_datetime(df_sample['date_publish'])
df_sample = df_sample[df_sample['topic'] != -1]  

# Count articles per topic per day
daily_ts = (
    df_sample.groupby([pd.Grouper(key='date_publish', freq='D'), 'topic'])
    .size()
    .unstack(fill_value=0)
)

# THE TOPIC MAP 
topic_category_map = {
    0: "narrative_protest_outcome",  
    1: "narrative_gov",    
    2: "narrative_dem_reform",
    3: "narrative_global",
    4: "narrative_religion",
    5: "narrative_elections",
    6: "narrative_basic_needs",
    7: "narrative_coup",
    8: "narrative_violence"
}

# Rename columns and save (Pointing to data folder)
daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

daily_ts.to_csv(os.path.join(data_dir, "var_input_ready_en.csv"))
print("\n--- FINAL VAR INPUT READY ---")
print(daily_ts.tail())