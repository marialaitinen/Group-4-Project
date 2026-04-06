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

# Setup NLTK French Stopwords
nltk.download('stopwords', quiet=True)
stop_words_fr = stopwords.words('french')

# Load your local French CSV from the data folder
print(f"Loading your French articles from: {data_dir}")
input_path = os.path.join(data_dir, "tagged_grievance_data_fr.csv")
df = pd.read_csv(input_path)

# Data Cleaning for the Model
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling (increased to 10k for consistency across languages)
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()
domain = df_sample["url"].tolist()

print(f"Training BERTopic on {len(docs)} articles...")

# Initialize Vectorizer and Multilingual Model
vectorizer = CountVectorizer(stop_words=stop_words_fr)
multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# THE SEED TOPICS (French)
seed_topic_list = [
    # 0. THE OUTCOME VARIABLE (Protests/Action)
    ["manifestation", "protestation", "émeutes", "grève"], 
    
    # 1. GOVERNMENT
    ["gouvernement", "politique", "anti-gouvernement", "anti-président", "président", "coalition", "opposition", "démission", "démissionne", "destitution", "impeachment"], 
    
    # 2. DEM-REFORM
    ["Printemps arabe", "journaliste", "journalistes", "liberté", "avocat", "démocratie", "place Tahrir", "loi", "droit", "indépendance", "anti-police", "constitution", "anti-corruption", "corruption", "réforme", "anti-ségrégation", "constitutionnel", "suffrage", "droit de vote", "femmes", "référendum", "fraude", "société civile"], 
    
    # 3. GLOBAL
    ["occuper", "occupation", "anti-WEF", "anti-Forum économique mondial", "anti-Davos", "anti-ONU", "anti-Nations unies", "anti-américain", "anti-États-Unis", "intervention", "étranger", "extérieur", "anti-mondialisation", "G20", "climat", "environnement", "environnemental", "immigration", "Brexit", "migration", "migrant", "réfugié", "droits humains", "droits de l'homme", "sommet", "anti-guerre"], 
    
    # 4. RELIGIOUS
    ["anti-blasphème", "mosquée", "Coran"], 
    
    # 5. ELECTIONS
    ["candidats", "vote", "voter", "électoral", "sondage", "scrutin"], 
    
    # 6. BASIC NEEDS
    ["anti-austérité", "austérité", "électricité", "énergie", "gilets jaunes", "gaz", "syndicat", "soins de santé", "santé", "éducation", "école", "terre", "agriculture"], 
    
    # 7. COUP
    ["évincé", "destitué", "assassinat", "assassiné", "militaire", "armée"], 
    
    # 8. VIOLENCE
    ["mortel", "violent", "guerre civile", "en feu", "incendie"] 
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

# Save Narrative Over Time (Pointing to data folder)
print("\nMapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_fr.csv"))

print("\nSUCCESS! 'narrative_pulse_data_fr.csv' is ready.")

# Save classified articles to the data folder
df_sample.to_csv(os.path.join(data_dir, "classified_articles_fr.csv"), index=False)

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

# Rename columns and save into the data folder
daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

daily_ts.to_csv(os.path.join(data_dir, "var_input_fr.csv")) 
print("\n--- FINAL VAR INPUT READY (FRENCH) ---")
print(daily_ts.tail())