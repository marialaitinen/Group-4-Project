from datasets import load_dataset
import pandas as pd
import csv
import re
import os

# --- CONFIGURATION ---
LANG = "zh" # Using 'zh' for Chinese in this dataset
REPO = "intfloat/multilingual_cc_news"

SIZE_LIMIT_GB = 1.0 
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# Cantonese Keywords (Outcome + Grievances)
KEYWORDS = [
    "抗議", "暴動", "罷工", "騷亂", "佔領運動",
    "政治", "反政府", "政府", "反總統", "總統", "聯盟", "反對派", "辭職", "彈劾",
    "記者", "自由", "律師", "民主", "法律", "獨立", "反警察", "憲法", "反腐敗", "腐敗", "改革", 
    "反種族隔離", "選舉權", "婦女", "全民公投", "詐欺", "公民社會",
    "世界經濟論壇", "達沃斯", "聯合國", "美國", "幹預", "外國", "反全球化", "二十國集團", 
    "氣候", "環境", "環保", "移民", "英國脫歐", "難民", "人權", "高峰會", "反戰",
    "反褻瀆", "清真寺", "古蘭經", "候選人", "投票", "選舉", "民調",
    "反緊縮", "緊縮", "電力", "能源", "黃背心運動", "天然氣", "工會", "醫療保健", "教育", 
    "學校", "土地", "農業", "被驅逐", "暗殺", "遇刺身亡", "軍事",
    "致命", "暴力", "內戰", "燃燒"
]

# FOR CHINESE: We remove the \b word boundaries
keyword_pattern = re.compile(r'(' + '|'.join(KEYWORDS) + r')', re.IGNORECASE)

def run_extraction():
    print(f" Starting 1GB extraction for Cantonese/Chinese...")
    
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
        first_row = next(iterator)
        keys = list(first_row.keys())
        
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        count = 0
        matches = 0
        
        for row in iterator:
            # Check content for keywords
            content = f"{row.get('title', '')} {row.get('maintext', '')}"
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

    print(f"\n SUCCESS! Chinese data saved to: {filename}")

run_extraction()