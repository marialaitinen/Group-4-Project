import pandas as pd
from urllib.parse import urlparse
import os

# 1. The Master Country Dictionary
tld_dictionary = {
    # ENGLISH
    'uk': 'United Kingdom', 'au': 'Australia', 'nz': 'New Zealand', 
    'za': 'South Africa', 'ie': 'Ireland', 'in': 'India', 'ng': 'Nigeria', 'ph': 'Philippines',
    
    # SPANISH
    'es': 'Spain', 'mx': 'Mexico', 'ar': 'Argentina', 'co': 'Colombia', 
    'cl': 'Chile', 'pe': 'Peru', 've': 'Venezuela', 'ec': 'Ecuador', 
    'gt': 'Guatemala', 'cu': 'Cuba', 'bo': 'Bolivia', 'do': 'Dominican Republic', 
    'hn': 'Honduras', 'py': 'Paraguay', 'sv': 'El Salvador', 'ni': 'Nicaragua', 
    'cr': 'Costa Rica', 'pa': 'Panama', 'uy': 'Uruguay', 'pr': 'Puerto Rico',
    
    # FRENCH
    'fr': 'France', 'be': 'Belgium', 'ch': 'Switzerland', 'sn': 'Senegal', 
    'ci': 'Ivory Coast', 'cm': 'Cameroon', 'ht': 'Haiti', 'ml': 'Mali', 
    'bf': 'Burkina Faso', 'mg': 'Madagascar', 'ca': 'Canada', 
    
    # TURKISH
    'tr': 'Turkey', 'cy': 'Cyprus',
    
    # CANTONESE/CHINESE
    'hk': 'Hong Kong', 'tw': 'Taiwan', 'mo': 'Macau', 'cn': 'China', 'sg': 'Singapore'
}

def get_country_from_url(url):
    try:
        domain = urlparse(str(url)).netloc 
        tld = domain.split('.')[-1].lower() 
        
        if tld in tld_dictionary:
            return tld_dictionary[tld]
        elif tld in ['com', 'org', 'net', 'info', 'news']:
            return "Generic_or_US" 
        else:
            return "Undefined"
    except:
        return "Undefined"

# 2. Loop Through All Languages
languages = ['en', 'es', 'fr', 'tr', 'zh']

# Ensure the data folder exists (though your extractor should have made it)
os.makedirs("data", exist_ok=True)

for lang in languages:
    # --- UPDATED PATHS TO POINT TO THE DATA FOLDER ---
    input_file = os.path.join("data", f"filtered_grievance_data_{lang}.csv")
    output_file = os.path.join("data", f"tagged_grievance_data_{lang}.csv")
    
    # Check if the extracted data actually exists in the data folder
    if not os.path.exists(input_file):
        print(f"\n⚠️ Skipping {lang.upper()} - Could not find '{input_file}'")
        continue
        
    print(f"\n=========================================")
    print(f" Processing {lang.upper()} Dataset...")
    
    # Load data from data/ folder
    df = pd.read_csv(input_file)
    
    # Apply the tagger
    df['country'] = df['url'].apply(get_country_from_url)
    
    # Print summary
    print(f"--- Top 5 Origins for {lang.upper()} ---")
    print(df['country'].value_counts().head(5))
    
    # Save the new tagged dataset back into the data/ folder
    df.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

print("\n🎉 ALL LANGUAGES TAGGED SUCCESSFULLY IN DATA FOLDER!")