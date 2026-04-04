from datasets import load_dataset
import pandas as pd
import csv
import re

# --- 配置 ---
LANG = "zh"
REPO = "intfloat/multilingual_cc_news"
SIZE_LIMIT_GB = 999  
BYTE_LIMIT = SIZE_LIMIT_GB * 1024 * 1024 * 1024

# 中文关键词
KEYWORDS = [
    # 政治与政府
    "政治", "反政府", "政府", "反總統", "總統", "聯盟", "反對派",
    "辭職", "彈劾", "記者", "自由", "律師", "民主", "法律",
    "獨立", "反警察", "憲法", "反腐敗", "腐敗", "改革",
    "反種族隔離", "合憲", "選舉權", "全民公投", "詐欺", "公民社會",
    
    # 抗议行动
    "抗議", "示威", "游行", "暴動", "騷亂", "罷工", "佔領",
    
    # 国际议题
    "世界經濟論壇", "達沃斯", "聯合國", "干預", "外國勢力",
    "反全球化", "二十國集團", "氣候", "環境", "環保",
    "移民", "英國脫歐", "難民", "人權", "峰會",
    
    # 宗教
    "褻瀆", "清真寺", "古蘭經",
    
    # 选举
    "候選人", "投票", "選舉", "民調",
    
    # 经济
    "緊縮", "電力", "能源", "黃背心", "天然氣",
    "工會", "醫療", "教育", "土地", "農業",
    
    # 暴力冲突
    "暗殺", "軍事", "騷亂", "暴力", "內戰", "燃燒",
    "致命", "被驅逐"
]

keyword_pattern = re.compile('(' + '|'.join(KEYWORDS) + ')')

def run_extraction():
    print(f"开始采集中文新闻数据...")
    ds = load_dataset(REPO, LANG, split="train", streaming=True)
    
    filename = "filtered_grievance_data_zh.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        iterator = iter(ds)
        first_row = next(iterator)
        keys = list(first_row.keys())
        print(f"检测到列名: {keys}")

        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        
        count = 0
        matches = 0
        
        # 处理第一行
        content = f"{first_row.get('title', '')} {first_row.get('maintext', '')}"
        if keyword_pattern.search(content):
            writer.writerow(first_row)
            matches += 1
        count += 1

        for row in iterator:
            content = f"{row.get('title', '')} {row.get('maintext', '')}"
            
            if keyword_pattern.search(content):
                writer.writerow(row)
                matches += 1
            
            count += 1
            if count % 1000 == 0:
                current_size = f.tell() / (1024**3)
                print(f"已扫描: {count:,} | 匹配: {matches:,} | 文件: {current_size:.3f} GB", end='\r')
                
                if f.tell() >= BYTE_LIMIT:
                    print(f"\n已达到 {SIZE_LIMIT_GB}GB 上限，停止。")
                    break

    print(f"\n完成！共扫描 {count:,} 篇，匹配 {matches:,} 篇")
    print(f"已保存到 {filename}")

run_extraction()