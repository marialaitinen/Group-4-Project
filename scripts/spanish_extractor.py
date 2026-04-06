from datasets import load_dataset
import pandas as pd
import csv
import re
import os

# --- CONFIGURATION ---
LANG = "es" # Spanish
REPO = "intfloat/multilingual_cc_news"

SIZE_LIMIT_GB = 1.0 
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# Spanish Keywords
KEYWORDS = [
    "protesta", "manifestación", "disturbio", "huelga", "político", "política", 
    "antigubernamental", "gobierno", "anti-presidente", "presidente", "coalition", 
    "oposición", "dimisión", "dimite", "destitución", "periodista", "periodistas", 
    "libertad", "abogado", "democracia", "ley", "independencia", "anti-policía", 
    "constitución", "anti-corrupción", "corrupción", "reforma", "anti-segregación", 
    "constitucional", "sufragio", "mujeres", "referéndum", "fraude", "sociedad civil", 
    "ocupar", "anti-globalización", "G20", "clima", "medioambiente", "inmigración", 
    "migración", "migrante", "refugiado", "derechos humanos", "cumbre", "antiguerra", 
    "mezquita", "Corán", "candidatos", "votar", "vota", "electoral", "encuesta", 
    "antiausteridad", "austeridad", "electricidad", "energia", "gas", "sindicato", 
    "asistencia sanitaria", "educación", "escuela", "tierra", "agricultura", 
    "derrocado", "magnicidio", "asesinado", "militar", "mortal", "violento", 
    "guerra civil", "quema"
]

# Compile for speed (using word boundaries \b for Spanish)
keyword_pattern = re.compile(r'\b(' + '|'.join(KEYWORDS) + r')\b', re.IGNORECASE)

def run_extraction():
    print(f" Starting 1GB Spanish extraction...")
    
    # --- BULLETPROOF PATH LOGIC ---
    # Gets the absolute path of the folder containing THIS script
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(base_dir, "data")
    
    # Ensure the 'data' folder exists
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, f"filtered_grievance_data_{LANG}.csv")
    print(f" Target path: {filename}")

    ds = load_dataset(REPO, LANG, split="train", streaming=True)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        iterator = iter(ds)
        first_row = next(iterator)
        keys = list(first_row.keys())
        
        # Identify columns
        text_col = next((k for k in ['text', 'content', 'maintext', 'body'] if k in keys), None)
        title_col = next((k for k in ['title', 'headline'] if k in keys), None)
        
        if not text_col:
            raise KeyError(f"Could not find a text column.")

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        count = 0
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

    print(f"\n SUCCESS! File is located at: {os.path.abspath(filename)}")

run_extraction()