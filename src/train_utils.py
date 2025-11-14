import pandas as pd
from tqdm.notebook import tqdm

def build_feature_dataframe(df_raw, feature_extractor_func, is_test_set=False):
    """
    Applica una funzione di estrazione feature specifica a un DataFrame.
    """
    print(f"Applicazione di '{feature_extractor_func.__name__}' a {len(df_raw)} righe...")
    battles_list = df_raw.to_dict('records')
    
    # Applica la funzione passata come argomento
    feature_rows = [feature_extractor_func(battle) for battle in tqdm(battles_list)]
    
    X_final = pd.DataFrame(feature_rows)
    X_final = X_final.fillna(0) # Fillna finale
    
    # Assicura che le colonne siano nell'ordine corretto
    X_final = X_final.reindex(sorted(X_final.columns), axis=1)
    
    print(f"Create {len(X_final.columns)} feature totali.")
    
    if is_test_set:
        return X_final, None
    else:
        y_final = df_raw['player_won'].astype(int)
        return X_final, y_final