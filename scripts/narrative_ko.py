import pandas as pd
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# --- PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Setup Stopwords
nltk.download('stopwords', quiet=True)
try:
    stop_words_list = stopwords.words('korean')
except:
    stop_words_list = []

# Load Input (Updated to use sample_*.csv)
input_path = os.path.join(data_dir, "sample_ko.csv")
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit()

df = pd.read_csv(input_path)
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

print(f"Training BERTopic on {len(docs)} Korean articles...")

# Run BERTopic with Translated Seeds
seed_topic_list = [['시위', '집회', '파업'], ['정부', '정치'], ['민주주의', '개혁'], ['국제', '글로벌'], ['종교', '교회'], ['선거', '투표'], ['경제', '가격'], ['쿠데타', '군사'], ['폭력', '전쟁']]
vectorizer = CountVectorizer(stop_words=stop_words_list)
multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

topic_model = BERTopic(
    language="multilingual", 
    embedding_model=multilingual_model,
    min_topic_size=20, 
    verbose=True,
    vectorizer_model=vectorizer,
    seed_topic_list=seed_topic_list 
)

topics, probs = topic_model.fit_transform(docs)
df_sample["topic"] = topics 

# Save Pulse Data
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_ko.csv"))
df_sample.to_csv(os.path.join(data_dir, "classified_articles_ko.csv"), index=False)

# Format for VAR
df_sample = df_sample[df_sample['topic'] != -1]  
daily_ts = (
    df_sample.groupby([pd.Grouper(key='date_publish', freq='D'), 'topic'])
    .size()
    .unstack(fill_value=0)
)

topic_category_map = {
    0: "narrative_protest_outcome", 1: "narrative_gov", 2: "narrative_dem_reform",
    3: "narrative_global", 4: "narrative_religion", 5: "narrative_elections",
    6: "narrative_basic_needs", 7: "narrative_coup", 8: "narrative_violence"
}

daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

daily_ts.to_csv(os.path.join(data_dir, "var_input_ko.csv")) 
print(f"--- FINAL VAR INPUT READY (KO) ---")
