import pandas as pd
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import jieba

# --- FIXING THE PARALLEL PATH ISSUE ---
# 1. Get the path to the 'scripts' folder
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go UP one level and then into the 'data' folder
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Custom Tokenizer for Chinese (Required because Chinese doesn't use spaces)
def tokenize_zh(text):
    return jieba.lcut(str(text))

# Load Data from the parallel data folder
print(f"Loading Cantonese articles from: {data_dir}")
input_path = os.path.join(data_dir, "tagged_grievance_data_zh.csv")
df = pd.read_csv(input_path)

# Data Cleaning for the Model
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling (set to 10k for consistency)
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

# Setup Vectorizer and Model
vectorizer = CountVectorizer(tokenizer=tokenize_zh)
multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Seed Topics (Cantonese/Traditional Chinese)
seed_topic_list = [
    ["抗議", "暴動", "罷工", "騷亂", "佔領運動"], # 0: Outcome
    ["政治", "政府", "總統", "辭職", "彈劾", "反對派"], # 1: Gov
    ["記者", "律師", "民主", "法律", "憲法", "腐敗", "改革", "公民社會"], # 2: Dem-Reform
    ["世界經濟論壇", "聯合國", "美國", "外國", "環境", "人權", "反戰"], # 3: Global
    ["清真寺", "古蘭經", "反褻瀆"], # 4: Religious
    ["候選人", "投票", "選舉", "民調"], # 5: Elections
    ["緊縮", "電力", "能源", "工會", "醫療", "教育", "農業"], # 6: Basic Needs
    ["被驅逐", "暗殺", "遇刺身亡", "軍事"], # 7: Coup
    ["致命", "暴力", "內戰", "燃燒"] # 8: Violence
]

# Run BERTopic
topic_model = BERTopic(
    language="multilingual",
    embedding_model=multilingual_model,
    vectorizer_model=vectorizer,
    seed_topic_list=seed_topic_list,
    min_topic_size=20,
    verbose=True
)

topics, _ = topic_model.fit_transform(docs)
df_sample = df_sample.copy()
df_sample["topic"] = topics

# Save classified articles back to the data folder
df_sample.to_csv(os.path.join(data_dir, "classified_articles_zh.csv"), index=False)

# Format for VAR
df_sample = df_sample[df_sample['topic'] != -1]
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

# Rename columns and save to the data folder
daily_ts = daily_ts.rename(columns=topic_category_map)
# Ensure we only keep columns that actually exist in the data
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts[columns_to_keep].to_csv(os.path.join(data_dir, "var_input_zh.csv"))

print("\nSUCCESS! 'var_input_zh.csv' is ready in the data folder.")
print(daily_ts.tail())