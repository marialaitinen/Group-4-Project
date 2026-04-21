from datasets import load_dataset
import pandas as pd
import csv
import os

REPO = "intfloat/multilingual_cc_news"
SIZE_LIMIT_GB = 5.0
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
    'ru': 'Russian',
    'tl': 'Tagalog',
    'zh': 'Cantonese',
    'bn': 'Bengali',
    'si': 'Sinhala',
    'ta': 'Tamil',
    'vi': 'Vietnamese',
    'af': 'Afrikaans',
    'ky': 'Kyrgyz',
    'ca': 'Catalan',
}

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

for lang in LANGUAGES:
    print(f"\nProcessing {lang.upper()}")
    filename = os.path.join(data_dir, f"sample_{lang}.csv")

    try:
        ds = load_dataset(REPO, lang, split="train", streaming=True)

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            iterator = iter(ds)
            first_row = next(iterator)
            keys = list(first_row.keys())

            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerow(first_row)

            count = 1
            for row in iterator:
                writer.writerow(row)
                count += 1
                if count % 5000 == 0:
                    current_size = f.tell() / (1024**3)
                    print(f"  {count:,} articles | {current_size:.2f} GB", end='\r')
                    if f.tell() >= BYTE_LIMIT:
                        print(f"\n  Reached {SIZE_LIMIT_GB}GB limit.")
                        break

        print(f"Done {lang.upper()}: {count} articles.")

    except Exception as e:
        print(f"Failed for {lang}: {e}")

#the summary
print("\n========== ARTICLE COUNT PER LANGUAGE ==========")
print(f"{'Language':<15} {'Code':<8} {'Articles':>10}")
print("-" * 35)

total = 0
for code, language in LANGUAGES.items():
    path = os.path.join(data_dir, f"sample_{code}.csv")
    if os.path.exists(path):
        count = len(pd.read_csv(path))
        total += count
        print(f"{language:<15} {code:<8} {count:>10}")
    else:
        print(f"{language:<15} {code:<8} {'NOT FOUND':>10}")

print("-" * 35)
print(f"{'TOTAL':<23} {total:>10}")
print("=" * 35)
