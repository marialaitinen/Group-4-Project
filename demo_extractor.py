from datasets import load_dataset
import pandas as pd

# We use the same 5 languages for your UvA project
langs = ["en", "es", "tr", "zh", "fr"]

for lang in langs:
    print(f" Starting download for: {lang}...")
    
    # NOTICE: No 'trust_remote_code' here. 
    # Version 2.15.0 doesn't need it!
    ds = load_dataset(
        "intfloat/multilingual_cc_news", 
        lang, 
        split="train", 
        streaming=True
    )
    
    # Grab the first 100 rows for your demo pipeline
    print(f"    Streaming rows...")
    demo_items = list(ds.take(100))
    
    # Save to CSV
    df = pd.DataFrame(demo_items)
    df.to_csv(f"demo_{lang}.csv", index=False)
    print(f" Success! 'demo_{lang}.csv' is ready.\n")

print(" All 5 language files have been created!")