from datasets import load_dataset
import pandas as pd
import csv
import re

# --- CONFIGURATION ---
LANG = "en"
REPO = "intfloat/multilingual_cc_news"

SIZE_LIMIT_GB = 1.0 
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# Your Keywords
KEYWORDS = [
    "political", "anti-government", "government", "anti-president", "president", 
    "coalition", "opposition", "resignation", "resigns", "impeachment", "protest", 
    "journalist", "journalists", "freedom", "lawyer", "democracy", "riot", "law", 
    "independence", "anti-police", "constitution", "anti-corruption", "corruption", 
    "reform", "anti-segregation", "constitutional", "suffrage", "women", "referendum", 
    "fraud", "civil society", "occupy", "WEF", "Davos", "anti-U.N.", "anti-US", "intervention", 
    "foreign", "anti-globalization", "G20", "climate", "environment", "environmental", 
    "immigration", "Brexit", "migration", "migrant", "refugee", "human rights", 
    "summit", "war", "blasphemy", "Mosque", "Quran", "candidates", "vote", "electoral", 
    "poll", "anti-austerity", "austerity", "electricity", "energy", "yellow vests", 
    "gas", "strike", "union", "healthcare", "education", "school", "land", 
    "agriculture", "ousted", "assassination", "assassinated", "military", "deadly", 
    "riots", "violent", "civil war", "burning"
]

# Compile for speed
keyword_pattern = re.compile(r'\b(' + '|'.join(KEYWORDS) + r')\b', re.IGNORECASE)

def run_extraction():
    print(f" Starting 1GB English extraction...")
    ds = load_dataset(REPO, LANG, split="train", streaming=True)
    
    filename = "filtered_grievance_data_en.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        iterator = iter(ds)
        
        # 
        first_row = next(iterator)
        keys = list(first_row.keys())
        print(f" Detected columns: {keys}")
        
        # Find which column holds the main article text
        # It's usually 'text', 'content', 'maintext', or 'body'
        text_col = next((k for k in ['text', 'content', 'maintext', 'body'] if k in keys), None)
        title_col = next((k for k in ['title', 'headline'] if k in keys), None)
        
        if not text_col:
            raise KeyError(f"Could not find a text column in {keys}. Please check the column names.")

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(first_row) # Don't lose the first row!
        
        count = 1
        matches = 1 if keyword_pattern.search(f"{first_row.get(title_col, '')} {first_row[text_col]}") else 0
        
        for row in iterator:
            # Safely combine title and text based on what we discovered
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

run_extraction()