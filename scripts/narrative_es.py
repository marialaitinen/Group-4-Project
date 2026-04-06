import pandas as pd
from bertopic import BERTopic
import os
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# --- FIXING THE PARALLEL PATH ISSUE ---
# 1. Get the path to the 'scripts' folder
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go UP one level and then into the 'data' folder
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Setup NLTK Spanish Stopwords
nltk.download('stopwords', quiet=True)
stop_words_es = stopwords.words('spanish')

# Load your local Spanish CSV from the data folder
print(f"Loading your Spanish articles from: {data_dir}")
input_path = os.path.join(data_dir, "tagged_grievance_data_es.csv")
df = pd.read_csv(input_path)

# Data Cleaning for the Model
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()
domain = df_sample["url"].tolist()

print(f"Training BERTopic on {len(docs)} articles...")

# Initialize Vectorizer and Multilingual Model
vectorizer = CountVectorizer(stop_words=stop_words_es)
multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# THE SEED TOPICS (Spanish)
seed_topic_list = [
    # 0. THE OUTCOME VARIABLE (Protests/Action)
    ["protesta", "manifestación", "disturbio", "huelga"], 
    # 1. GOVERNMENT
    ["político", "política", "antigubernamental", "gobierno", "anti-presidente", "presidente", "coalición", "opposition", "dimisión", "dimite", "destitución"], 
    # 2. DEM-REFORM
    ["Primavera Árabe", "periodista", "periodistas", "libertad", "abogado", "democracia", "plaza de la Liberación", "plaza Tahrir", "ley", "independencia", "anti-policía", "constitución", "anti-corrupción", "corrupción", "reforma", "anti-segregación", "constitucional", "sufragio", "mujeres", "referéndum", "fraude", "sociedad civil"], 
    # 3. GLOBAL
    ["ocupar", "anti WEF", "anti Davos", "anti-Unión Europea", "anti-Estados Unidos", "intervención", "extranjero", "extranjera", "antiglobalización", "G20", "clima", "medioambiente", "medioambiental", "inmigración", "migración del Brexit", "migrante", "refugiado", "refugiada", "derechos humanos", "cumbre", "antiguerra"], 
    # 4. RELIGIOUS
    ["antiblasfemia", "mezquita", "Corán"], 
    # 5. ELECTIONS
    ["candidatos", "candidatas", "votar", "vota", "electoral", "encuesta"], 
    # 6. BASIC NEEDS
    ["antiausteridad", "austeridad", "electricidad", "energia", "chalecos amarillos", "gas", "sindicato", "asistencia sanitaria", "educación", "escuela", "tierra", "agricultura"], 
    # 7. COUP
    ["derrocado", "derrocada", "magnicidio", "asesinado", "asesinada", "militar"], 
    # 8. VIOLENCE
    ["mortal", "violento", "violenta", "guerra civil", "quema"] 
]

# Run BERTopic with Multilingual Settings
topic_model = BERTopic(
    language="multilingual", 
    embedding_model=multilingual_model,
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
print("\n--- TOP NARRATIVES IDENTIFIED ---")
print(topic_info[['Topic', 'Count', 'Name']].head(10))

# Save Narrative Over Time (Pointing to parallel data folder)
print("\nMapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_es.csv"))

print("\nSUCCESS! 'narrative_pulse_data_es.csv' is ready.")

# Save classified articles (Pointing to parallel data folder)
df_sample.to_csv(os.path.join(data_dir, "classified_articles_es.csv"), index=False)

# Format for VAR Model
df_sample['date_publish'] = pd.to_datetime(df_sample['date_publish'])
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

# Rename columns and save (Pointing to parallel data folder)
daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

daily_ts.to_csv(os.path.join(data_dir, "var_input_es.csv")) 
print("\n--- FINAL VAR INPUT READY (SPANISH) ---")
print(daily_ts.tail())