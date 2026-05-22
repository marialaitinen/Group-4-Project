from datasets import load_dataset
import pandas as pd
import csv
import os

REPO = "intfloat/multilingual_cc_news"
SIZE_LIMIT_GB = 1
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

LANGUAGES = {
    'tr': 'Turkish',
    'id': 'Indonesian',
    'bg': 'Bulgarian',
    'cs': 'Czech',
    'de': 'German',
    'el': 'Greek',
    'hu': 'Hungarian',
    'it': 'Italian',
    'pl': 'Polish',
    'ko': 'Korean',
    'zh': 'Cantonese',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'vi': 'Vietnamese',
    'ca': 'Catalan',
    'ky': 'Kyrgyz',
}

KEYWORDS = {
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
    'bn': [['বিক্ষোভ', 'সমাবেশ', 'ধর্মঘট'], ['সরকার', 'রাজনীতি'], ['গণতন্ত্র', 'সংস্কার'], ['আন্তর্জাতিক', 'বিশ্বব্যাপী'], ['ধর্ম', 'মসजीদ'], ['নির্বাচন', 'ভোট'], ['অর্থনীতি', 'দাম'], ['অভ্যوت্থان', 'সামরিক'], ['সহিংসতা', 'যুদ্ধ']],
    'vi': [['biểu tình', 'biểu dương', 'đình công'], ['chính phủ', 'chính trị'], ['dân chủ', 'cải cách'], ['quốc tế', 'toàn cầu'], ['tôn giáo', 'nhà thờ'], ['bầu cử', 'bỏ phiếu'], ['kinh tế', 'giá cả'], ['đảo chính', 'quân sự'], ['bạo lực', 'chiến tranh']],
    'ca': [['protesta', 'manifestació', 'vaga'], ['govern', 'política'], ['democràcia', 'reforma'], ['internacional', 'global'], ['religió', 'església'], ['eleccions', 'vot'], ['economia', 'preus'], ["cop d'estat", 'militar'], ['violència', 'guerra']],
    'ky': [['митинг', 'нааразычылык', 'стачка'], ['өкмөт', 'саясат'], ['демократия', 'реформа'], ['эл аралык', 'глобалдык'], ['дин', 'мечит'], ['шайлоо', 'добуш'], ['экономика', 'баалар'], ['төңкөрүш', 'аскердик'], ['зомбулук', 'согуш']]
}

# Path structure configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# Keeps track of real-time metrics safely without reading huge CSV files into memory
article_counts = {lang: 0 for lang in LANGUAGES}

for lang in LANGUAGES:
    print(f"\nProcessing {lang.upper()}")
    filename = os.path.join(data_dir, f"sample_{lang}.csv")
    
    # Flatten the list of lists into a single 1D list and force everything to lowercase
    nested_keywords = KEYWORDS.get(lang, [])
    lang_keywords = [word.lower() for sublist in nested_keywords for word in sublist]
    
    if not lang_keywords:
        print(f"  Skipping {lang.upper()}: No keywords provided.")
        continue

    try:
        # Load stream
        ds = load_dataset(REPO, lang, split="train", streaming=True)
        
        # Safely pull exact file header keys directly from dataset schema metadata
        keys = list(ds.features.keys())

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

            count = 0
            for row in ds:
                # Merge title and body text into a clean lowercase searchable string
                article_content = f"{row.get('title', '')} {row.get('text', '')}".lower()
                
                # Write to file if any localized keyword matches the text
                if any(keyword in article_content for keyword in lang_keywords):
                    writer.writerow(row)
                    count += 1
                    
                    # Size checker updates every 1,000 matches to keep I/O fast
                    if count % 1000 == 0:
                        current_size = f.tell() / (1024**3)
                        print(f"  {count:,} matching articles | {current_size:.2f} GB", end='\r')
                        
                        if f.tell() >= BYTE_LIMIT:
                            print(f"\n  Reached maximum size threshold of {SIZE_LIMIT_GB}GB.")
                            break
                            
            article_counts[lang] = count

        print(f"Done {lang.upper()}: {count:,} matched articles saved.")

    except Exception as e:
        print(f"Failed processing pipeline for {lang}: {e}")

# =====================================================================
# THE SUMMARY 
# =====================================================================
print("\n========== ARTICLE COUNT PER LANGUAGE ==========")
print(f"{'Language':<15} {'Code':<8} {'Articles':>10}")
print("-" * 35)

total = 0
for code, language in LANGUAGES.items():
    count = article_counts.get(code, 0)
    path = os.path.join(data_dir, f"sample_{code}.csv")
    
    if os.path.exists(path):
        total += count
        print(f"{language:<15} {code:<8} {count:>10,}")
    else:
        print(f"{language:<15} {code:<8} {'NOT FOUND':>10}")

print("-" * 35)
print(f"{'TOTAL':<23} {total:>10,}")
print("=" * 35)