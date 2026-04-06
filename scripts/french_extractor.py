from datasets import load_dataset
import pandas as pd
import csv
import re
import os

# --- CONFIGURATION ---
LANG = "fr"
REPO = "intfloat/multilingual_cc_news"

SIZE_LIMIT_GB = 1.0 
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# Your French Keywords
KEYWORDS = [
    "manifestation", "protestation", "émeutes", "grève",
    "gouvernement", "politique", "anti-gouvernement", "anti-président", "président", 
    "coalition", "opposition", "démission", "démissionne", "destitution", "impeachment", 
    "Printemps arabe", "journaliste", "journalistes", "liberté", "avocat", "démocratie", 
    "place Tahrir", "loi", "droit", "indépendance", "anti-police", "constitution", 
    "anti-corruption", "corruption", "réforme", "anti-ségrégation", "constitutionnel", 
    "suffrage", "droit de vote", "femmes", "référendum", "fraude", "société civile", 
    "occuper", "occupation", "anti-WEF", "anti-Forum économique mondial", "anti-Davos", 
    "anti-ONU", "anti-Nations unies", "anti-américain", "anti-États-Unis", "intervention", 
    "étranger", "extérieur", "anti-mondialisation", "G20", "climat", "environnement", 
    "environnemental", "immigration", "Brexit", "migration", "migrant", "réfugié", 
    "droits humains", "droits de l'homme", "sommet", "anti-guerre", 
    "anti-blasphème", "mosquée", "Coran", 
    "candidats", "vote", "voter", "électoral", "sondage", "scrutin", 
    "anti-austérité", "austérité", "électricité", "énergie", "gilets jaunes", "gaz", 
    "syndicat", "soins de santé", "santé", "éducation", "école", "terre", "agriculture", 
    "évincé", "destitué", "assassinat", "assassiné", "militaire", "armée", 
    "mortel", "violent", "guerre civile", "en feu", "incendie"
]

# Compile for speed
keyword_pattern = re.compile(r'\b(' + '|'.join(KEYWORDS) + r')\b', re.IGNORECASE)

def run_extraction():
    print(f" Starting 1GB French extraction...")
    
    # --- BULLETPROOF PATH LOGIC ---
    # Ensures the 'data' folder is created exactly where this script lives
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
        
        # Initialize counts
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

    print(f"\n SUCCESS! French data saved to: {filename}")

run_extraction()