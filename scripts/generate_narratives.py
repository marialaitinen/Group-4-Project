import os

# Your final 16 languages
LANGUAGES = {
    'tr': 'Turkish', 'id': 'Indonesian', 'bg': 'Bulgarian', 'cs': 'Czech', 
    'de': 'German', 'el': 'Greek', 'hu': 'Hungarian', 'it': 'Italian', 
    'pl': 'Polish', 'ko': 'Korean', 'zh': 'Chinese', 'bn': 'Bengali', 
    'ta': 'Tamil', 'vi': 'Vietnamese', 'ca': 'Catalan', 'ky': 'Kyrgyz'
}

# Native Seed Topics for Grounding
TRANSLATED_SEEDS = {
    'tr': [['protesto', 'gösteri', 'grev'], ['hükümet', 'siyasi'], ['demokrasi', 'reform'], ['küresel', 'uluslararası'], ['din', 'cami'], ['seçim', 'oy'], ['ekonomi', 'temel ihtiyaçlar'], ['darbe', 'askeri'], ['şiddet', 'iç savaş']],
    'id': [['protes', 'demonstrasi', 'pemogokan'], ['pemerintah', 'politik'], ['demokrasi', 'reformasi'], ['internasional', 'global'], ['agama', 'masjid'], ['pemilu', 'suara'], ['ekonomi', 'harga'], ['kudeta', 'militer'], ['kekerasan', 'perang']],
    'bg': [['протест', 'митинг', 'стачка'], ['правителство', 'политика'], ['демокрация', 'реформа'], ['международен', 'глобален'], ['религия', 'църква'], ['избори', 'вот'], ['икономика', 'цени'], ['преврат', 'военен'], ['насилие', 'война']],
    'cs': [['protest', 'demonstrace', 'stávka'], ['vláda', 'politika'], ['demokracie', 'reforma'], ['mezinárodní', 'globální'], ['náboženství', 'církev'], ['volby', 'hlasování'], ['ekonomika', 'ceny'], ['převrat', 'vojenský'], ['nasilí', 'válka']],
    'de': [['protest', 'demonstration', 'streik'], ['regierung', 'politik'], ['demokratie', 'reform'], ['international', 'global'], ['religion', 'kirche'], ['wahl', 'stimme'], ['wirtschaft', 'preise'], ['putsch', 'militär'], ['gewalt', 'krieg']],
    'el': [['διαμαρτυρία', 'διαδήλωση', 'απεργία'], ['κυβέρνηση', 'πολιτική'], ['δημοκρατία', 'μεταρρύθμιση'], ['διεθνής', 'παγκόσμιος'], ['θρησκεία', 'εκκλησία'], ['εκλογές', 'ψήφος'], ['οικονομία', 'τιμές'], ['πραξικόπημα', 'στρατιωτικός'], ['βία', 'πόλεμος']],
    'hu': [['tüntetés', 'demonstráció', 'sztrájk'], ['kormány', 'politika'], ['demokrácia', 'reform'], ['nemzetközi', 'globális'], ['vallás', 'egyház'], ['választás', 'szavazás'], ['gazdaság', 'árak'], ['puccs', 'katonai'], ['erőszak', 'háború']],
    'it': [['protesta', 'dimostrazione', 'sciopero'], ['governo', 'politica'], ['democrazia', 'riforma'], ['internazionale', 'globale'], ['religione', 'chiesa'], ['elezione', 'voto'], ['economia', 'prezzi'], ['colpo di stato', 'militare'], ['violenza', 'guerra']],
    'pl': [['protest', 'demonstracja', 'strajk'], ['rząd', 'polityka'], ['demokracja', 'reforma'], ['międzynarodowy', 'globalny'], ['religia', 'kościół'], ['wyborze', 'głosowanie'], ['ekonomia', 'ceny'], ['pucz', 'wojskowy'], ['przemoc', 'wojna']],
    'ko': [['시위', '집회', '파업'], ['정부', '정치'], ['민주주의', '개혁'], ['국제', '글로벌'], ['종교', '교회'], ['선거', '투표'], ['경제', '가격'], ['쿠데타', '군사'], ['폭력', '전쟁']],
    'zh': [['抗議', '示威', '罷工'], ['政府', '政治'], ['民主', '改革'], ['國際', '全球'], ['宗教', '教會'], ['選舉', '投票'], ['經濟', '價格'], ['政變', '軍事'], ['暴力', '戰爭']],
    'bn': [['বিক্ষোভ', 'সমাবেশ', 'ধর্মঘট'], ['সরকার', 'রাজনীতি'], ['গণতন্ত্র', 'সংস্কার'], ['আন্তর্জাতিক', 'বিশ্বব্যাপী'], ['ধর্ম', 'মসজিদ'], ['নির্বাচন', 'ভোট'], ['অর্থনীতি', 'দাম'], ['অভ্যুত্থান', 'সামরিক'], ['সহিংসতা', 'যুদ্ধ']],
    'ta': [['எதிர்ப்பு', 'ஆர்ப்பாட்டம்', 'வேலைநிறுத்தம்'], ['அரசாங்கம்', 'அரசியல்'], ['ஜனநாயகம்', 'சீர்திருத்தம்'], ['சர்வதேச', 'உலகளாவிய'], ['மதம்', 'கோயில்'], ['தேர்தல்', 'வாக்கு'], ['பொருளாதார', 'விலைகள்'], ['சதி', 'இராணுவ'], ['வன்முறை', 'போர்']],
    'vi': [['biểu tình', 'biểu dương', 'đình công'], ['chính phủ', 'chính trị'], ['dân chủ', 'cải cách'], ['quốc tế', 'toàn cầu'], ['tôn giáo', 'nhà thờ'], ['bầu cử', 'bỏ phiếu'], ['kinh tế', 'giá cả'], ['đảo chính', 'quân sự'], ['bạo lực', 'chiến tranh']],
    'ca': [['protesta', 'manifestació', 'vaga'], ['govern', 'política'], ['democràcia', 'reforma'], ['internacional', 'global'], ['religió', 'església'], ['eleccions', 'vot'], ['economia', 'preus'], ["cop d'estat", 'militar'], ['violència', 'guerra']],
    'ky': [['митинг', 'нааразычылык', 'стачка'], ['өкмөт', 'саясат'], ['демократия', 'реформа'], ['эл аралык', 'глобалдык'], ['дин', 'мечит'], ['шайлоо', 'добуш'], ['экономика', 'баалар'], ['төңкөрүш', 'аскердик'], ['зомбулук', 'согуш']]
}

template = """import pandas as pd
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
    stop_words_list = stopwords.words('{nltk_lang}')
except:
    stop_words_list = []

# Load Input (Updated to use sample_*.csv)
input_path = os.path.join(data_dir, "sample_{lang_code}.csv")
if not os.path.exists(input_path):
    print(f"File not found: {{input_path}}")
    exit()

df = pd.read_csv(input_path)
df['date_publish'] = pd.to_datetime(df['date_publish'], errors='coerce')
df = df.dropna(subset=['maintext', 'date_publish'])

# Subsampling
df_sample = df.sample(n=min(10000, len(df)), random_state=42).sort_values('date_publish')
docs = df_sample['maintext'].tolist()
timestamps = df_sample['date_publish'].tolist()

print(f"Training BERTopic on {{len(docs)}} {lang_name} articles...")

# Run BERTopic with Translated Seeds
seed_topic_list = {seed_list}
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
topics_over_time.to_csv(os.path.join(data_dir, "narrative_pulse_data_{lang_code}.csv"))
df_sample.to_csv(os.path.join(data_dir, "classified_articles_{lang_code}.csv"), index=False)

# Format for VAR
df_sample = df_sample[df_sample['topic'] != -1]  
daily_ts = (
    df_sample.groupby([pd.Grouper(key='date_publish', freq='D'), 'topic'])
    .size()
    .unstack(fill_value=0)
)

topic_category_map = {{
    0: "narrative_protest_outcome", 1: "narrative_gov", 2: "narrative_dem_reform",
    3: "narrative_global", 4: "narrative_religion", 5: "narrative_elections",
    6: "narrative_basic_needs", 7: "narrative_coup", 8: "narrative_violence"
}}

daily_ts = daily_ts.rename(columns=topic_category_map)
columns_to_keep = [col for col in topic_category_map.values() if col in daily_ts.columns]
daily_ts = daily_ts[columns_to_keep]

daily_ts.to_csv(os.path.join(data_dir, "var_input_{lang_code}.csv")) 
print(f"--- FINAL VAR INPUT READY ({lang_upper}) ---")
"""

for code, name in LANGUAGES.items():
    seeds = TRANSLATED_SEEDS.get(code)
    file_content = template.format(
        lang_code=code, 
        lang_name=name, 
        lang_upper=code.upper(), 
        nltk_lang=name.lower(), 
        seed_list=seeds
    )
    
    filename = f"narrative_{code}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(file_content)
    print(f"Generated {filename}")

print("\nDone! Narrative scripts pointing to sample files are ready.")