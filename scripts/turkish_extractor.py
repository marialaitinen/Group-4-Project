from datasets import load_dataset
import pandas as pd
import csv
import re
import os

# --- CONFIGURATION ---
LANG = "tr"
REPO = "intfloat/multilingual_cc_news"

SIZE_LIMIT_GB = 1.0 
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# Your Turkish Keywords
KEYWORDS = [
    "grev", "isyanlar", "olaylar", "ayaklanmalar",
    "hükümet", "siyasi", "politik", "hükümet karşıtı", "muhalif", "cumhurbaşkanı karşıtı", 
    "başkan karşıtı", "cumhurbaşkanı", "başkan", "koalisyon", "muhalefet", "istifa", 
    "istifa etti", "istifa ediyor", "görevden alma", "azil", 
    "Arap Baharı", "gazeteci", "gazeteciler", "özgürlük", "avukat", "demokrasi", 
    "Tahrir Meydanı", "yasa", "hukuk", "bağımsızlık", "polis karşıtı", "anayasa", 
    "yolsuzluk karşıtı", "yolsuzluk", "reform", "ayrımcılık karşıtı", "ayrışma karşıtı", 
    "anayasal", "seçme hakkı", "oy hakkı", "seçme ve seçilme hakkı", "kadınlar", "kadın", 
    "referandum", "hile", "seçim hilesi", "dolandırıcılık", "sivil toplum", 
    "işgal et", "işgal etmek", "işgal", "Dünya Ekonomik Forumu karşıtı", "WEF karşıtı", 
    "Davos karşıtı", "BM karşıtı", "Birleşmiş Milletler karşıtı", "ABD karşıtı", "müdahale", 
    "yabancı", "dış", "küreselleşme karşıtı", "G20", "iklim", "çevre", "çevresel", 
    "çevreyle ilgili", "göç", "Brexit", "göç hareketi", "göçmen", "mülteci", "insan hakları", 
    "zirve", "savaş karşıtı", 
    "küfür karşıtı", "dine hakaret karşıtı", "cami", "Kuran", "Kur'an", 
    "adaylar", "oy", "oy vermek", "seçimle ilgili", "seçimsel", "anket", "yoklama", "sandık", 
    "kemer sıkma karşıtı", "kemer sıkma", "kemer sıkma politikası", "elektrik", "enerji", 
    "Sarı Yelekliler", "gaz", "akaryakıt", "sendika", "sağlık hizmetleri", "eğitim", "okul", 
    "toprak", "arazi", "tarım", 
    "görevden alındı", "devrildi", "suikast", "suikasta uğradı", "öldürüldü", "askeri", "ordu", 
    "ölümcül", "şiddet içeren", "şiddetli", "şiddet", "iç savaş", "yanma", "ateşe verme", 
    "yanan", "yakma"
]

# Compile for speed
keyword_pattern = re.compile(r'\b(' + '|'.join(KEYWORDS) + r')\b', re.IGNORECASE)

def run_extraction():
    print(f" Starting 1GB extraction for Turkish...")
    
    # --- BULLETPROOF PATH LOGIC ---
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True) 
    
    filename = os.path.join(data_dir, f"filtered_grievance_data_{LANG}.csv")
    print(f" Target path: {filename}")

    ds = load_dataset(REPO, LANG, split="train", streaming=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        iterator = iter(ds)
        
        # Detect columns dynamically
        first_row = next(iterator)
        keys = list(first_row.keys())
        print(f" Detected columns: {keys}")
        
        text_col = next((k for k in ['text', 'content', 'maintext', 'body'] if k in keys), None)
        title_col = next((k for k in ['title', 'headline'] if k in keys), None)
        
        if not text_col:
            raise KeyError(f"Could not find a text column.")

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        # Check first row for keywords
        count = 1
        first_content = f"{first_row.get(title_col, '')} {first_row[text_col]}"
        if keyword_pattern.search(first_content):
            writer.writerow(first_row)
            matches = 1
        else:
            matches = 0
        
        for row in iterator:
            content = f"{row.get(title_col, '')} {row[text_col]}"
            
            if keyword_pattern.search(content):
                writer.writerow(row)
                matches += 1
            
            count += 1
            if count % 2000 == 0:
                current_size = f.tell() / (1024**3)
                print(f" Scanned: {count:,} | Matched: {matches:,} | File: {current_size:.2f} GB", end='\r')
                
                if f.tell() >= BYTE_LIMIT:
                    print(f"\n Reached {SIZE_LIMIT_GB}GB limit.")
                    break

    print(f"\n SUCCESS! Turkish data saved to: {filename}")

run_extraction()