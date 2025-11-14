import json
import pandas as pd
import warnings
import os

# Ignora i warning per pulizia
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path, seed=123, is_train=True):
    """
    Carica, pulisce e fa lo shuffle dei dati da un file .jsonl.
    """
    print(f"Caricamento dati da: {os.path.basename(file_path)}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df_raw = pd.DataFrame(data)

    if not is_train:
        # Per il test set, non c'è pulizia da fare, restituisci subito
        print(f"Dati di test pronti. Shape: {df_raw.shape}")
        return df_raw

    # --- Pulizia (solo per training set, come da notebook) ---
    
    # 1. Rimozione riga corrotta
    ROW_TO_DROP = 4877
    if ROW_TO_DROP in df_raw.index:
        df_cleaned = df_raw.drop(index=ROW_TO_DROP)
    else:
        df_cleaned = df_raw.copy()

    # 2. Rimozione livelli non standard
    indices_non_standard_level = []
    for index, row in df_cleaned.iterrows():
        found_non_100 = False
        if isinstance(row['p1_team_details'], list):
            for pokemon in row['p1_team_details']:
                if isinstance(pokemon, dict) and pokemon.get('level') != 100:
                    found_non_100 = True; break
        if not found_non_100:
            p2_lead = row['p2_lead_details']
            if isinstance(p2_lead, dict) and p2_lead.get('level') != 100:
                found_non_100 = True
        if found_non_100:
            indices_non_standard_level.append(index)

    if indices_non_standard_level:
        df_cleaned = df_cleaned.drop(index=indices_non_standard_level)

    # --- Shuffle ---
    df_shuffled = df_cleaned.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Dati di training pronti. Shape: {df_shuffled.shape}")
    return df_shuffled