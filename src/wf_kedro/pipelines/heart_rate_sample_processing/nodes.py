"""
This is a boilerplate pipeline 'heart_rate_sample_processing'
generated using Kedro 1.3.1
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable
import tensorflow as tf

def process_parquet_partitions(partitioned_input: Dict[str, Callable[[], pd.DataFrame]]) -> pd.DataFrame:
    """Verwerkt alle Parquet partities en voegt ze samen tot één DataFrame."""
    dfs = []
    
    # Kedro geeft een dictionary met laad-functies per bestand
    for partition_id, load_func in partitioned_input.items():
        try:
            # Laad de partitie (Kedro leest het parquet-bestand hier in)
            df_part = load_func()
            
            # Verwijder kolommen direct als ze bestaan
            kolommen_te_verwijderen = ['id', 'is_manual', 'min', 'max', 'value', 'duration']
            df_part = df_part.drop(columns=[col for col in kolommen_te_verwijderen if col in df_part.columns])
            
            if 'heart_rate_samples' in df_part.columns:
                df_plat = df_part.explode('heart_rate_samples').dropna(subset=['heart_rate_samples'])
                
                if len(df_plat) > 0:
                    # De supersnelle NumPy vectorisatie
                    matrix = np.vstack(df_plat['heart_rate_samples'].values)
                    df_plat['time_offset_in_seconds'] = matrix[:, 0]
                    df_plat['bpm'] = matrix[:, 1]
                    
                    df_part = df_plat.drop(columns=['heart_rate_samples'])
                    dfs.append(df_part)
                    
        except Exception as e:
            print(f"Fout bij partitie {partition_id}: {e}")
            
    if not dfs:
        raise ValueError("Geen bruikbare data ingelezen uit de partities.")
        
    return pd.concat(dfs, ignore_index=True)


def create_keras_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    """Zet het opgeschoonde Pandas DataFrame om naar een Keras-ready tf.data.Dataset."""
    
    # Sorteer de data (optioneel, handig voor tijdreeksen)
    df = df.sort_values(by=['time', 'time_offset_in_seconds'])
    
    # Splits de data in features (inputs) en eventuele targets (labels)
    # Pas de kolomnamen aan op basis van wat jouw Keras model verwacht
    features = {
        'time_offset': df['time_offset_in_seconds'].values,
        'timezone_offset': df['timezone_offset_in_seconds'].values
        # Voeg hier andere dimensies/variabelen toe
    }
    labels = df['bpm'].values  # We voorspellen bijvoorbeeld de hartslag (BPM)
    
    # Maak de TensorFlow / Keras dataset aan
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # Voeg handige Keras optimalisaties toe (shuffelen, batching en prefeching)
    dataset = dataset.shuffle(buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return dataset

