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

# Setup NLTK Turkish Stopwords
nltk.download('stopwords', quiet=True)
stop_words_tr = stopwords.words('turkish')

# Load your local Turkish CSV from the data folder
print(f"Loading your Turkish articles from: {data_dir}")
input_path = os.path.join(data_dir, "tagged_grievance_data_tr.csv")
df = pd.read_csv(input_path)

# Data Cleaning for the Model
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling (Set to 10,000 for consistency)
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

print(f"Training BERTopic on {len(docs)} articles...")

# Initialize Vectorizer and Multilingual Model
vectorizer = CountVectorizer(stop_words=stop_words_tr)
multilingual_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# THE SEED TOPICS (Turkish)
seed_topic_list = [
    # 0. THE OUTCOME VARIABLE (Protests/Action)
    ["protesto", "gösteri", "grev", "isyanlar", "ayaklanmalar", "olaylar"], 
    
    # 1. GOVERNMENT
    ["hükümet", "siyasi", "politik", "hükümet karşıtı", "muhalif", "cumhurbaşkanı karşıtı", "başkan karşıtı", "cumhurbaşkanı", "başkan", "koalisyon", "muhalefet", "istifa", "istifa etti", "istifa ediyor", "görevden alma", "azil"], 
    
    # 2. DEM-REFORM
    ["Arap Baharı", "gazeteci", "gazeteciler", "özgürlük", "avukat", "demokrasi", "Tahrir Meydanı", "yasa", "hukuk", "bağımsızlık", "polis karşıtı", "anayasa", "yolsuzluk karşıtı", "yolsuzluk", "reform", "ayrımcılık karşıtı", "ayrışma karşıtı", "anayasal", "seçme hakkı", "oy hakkı", "seçme ve seçilme hakkı", "kadınlar", "kadın", "referandum", "hile", "seçim hilesi", "dolandırıcılık", "sivil toplum"], 
    
    # 3. GLOBAL
    ["işgal", "işgal etmek", "Dünya Ekonomik Forumu karşıtı", "WEF karşıtı", "Davos karşıtı", "BM karşıtı", "Birleşmiş Milletler karşıtı", "ABD karşıtı", "müdahale", "yabancı", "dış", "küreselleşme karşıtı", "G20", "iklim", "çevre", "çevresel", "çevreyle ilgili", "göç", "Brexit", "göç hareketi", "göçmen", "mülteci", "insan hakları", "zirve", "savaş karşıtı"], 
    
    # 4. RELIGIOUS
    ["küfür karşıtı", "dine hakaret karşıtı", "cami", "Kuran", "Kur'an"], 
    
    # 5. ELECTIONS
    ["adaylar", "oy", "oy vermek", "seçimle ilgili", "seçimsel", "anket", "yoklama", "sandık"], 
    
    # 6. BASIC NEEDS
    ["kemer sıkma karşıtı", "kemer sıkma", "kemer sıkma politikası", "elektrik", "enerji", "Sarı Yelekliler", "gaz", "akaryakıt", "sendika", "sağlık hizmetleri", "eğitim", "okul", "toprak", "arazi", "tarım"], 
    
    # 7. COUP
    ["görevden alındı", "devrildi", "suikast", "suikasta uğradı", "öldürüldü", "askeri", "ordu"], 
    
    # 8. VIOLENCE
    ["ölümcül", "şiddet içeren", "şiddetli", "şiddet", "iç savaş", "yanma", "ateşe verme", "yakma"] 
]

# Run BERTopic
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

# Narrative Summary
topic_info = topic_model.get_topic_info()
print("\n--- TOP NARRATIVES IDENTIFIED (TURKISH) ---")
print(topic_info[['Topic', 'Count', 'Name']].head(10))

# Save Over Time Data (Pointing to data folder)
print("\nMapping narratives to your timeline...")
topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_tr.csv"))

# Save classified articles (Pointing to data folder)
df_sample.to_csv(os.path.join(data_dir, "classified_articles_tr.csv"), index=False)

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

daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

# Save final input for VAR (Pointing to data folder)
daily_ts.to_csv(os.path.join(data_dir, "var_input_tr.csv")) 
print("\n--- FINAL VAR INPUT READY (TURKISH) ---")
print(daily_ts.tail())