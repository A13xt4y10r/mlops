
import numpy as np
from scipy import stats
from constants import TARGET_COLUMN_NAME


def create_new_features(df, base_col_name, orig_col_name, new_col_name):
    """Új jellemzők létrehozása

    Paraméterek:
    df -- DataFrame az adatokkal
    base_col_name -- a bázis oszlop neve
    orig_col_name -- az eredeti oszlop neve
    new_col_name -- az új oszlop neve

    Visszatérési értékek:
    df -- DataFrame az új jellemzőkkel
    """
    df[new_col_name] = (df[base_col_name] - df[orig_col_name]).dt.days

    # az eredeti oszlopok eldobása, csak a target marad
    df = df.drop(orig_col_name, axis=1)
    df = df.drop(base_col_name, axis=1)

    return df

def drop_unwanted_rows(df):
    # ezekben nullok vannak
    # ha nem dátum lenne, akkor a clear_date oszlopot meg lehet próbálni majd a nullokat átlag számokkal helyettesíteni
    # de mivel dátum, és relatív érték, így nincs átlaga, nem tudom a hiányzó értékeket feltölteni
    # az invoice_id-ből meg csak 6 hiányzik, nincs értelme feltölteni
    # ezért eltávolítom azokat a sorokat, ahol NaN érték van ebben a 2 oszlopban
    df = df.dropna(subset=['clear_date'])
    df = df.dropna(subset=['invoice_id'])

    # duplikált sorok törlése az összes oszlop alapján, csak az első megtartása, és az eredeti DataFrame frissítése
    df.drop_duplicates(keep='first', inplace=True)

    return df

def drop_outlayers(df):
    #outlayer kezelés
    # Azonosítsuk az outliereket a targeten
    z_scores = np.abs(stats.zscore(df[[TARGET_COLUMN_NAME]]))
    # NumPy tömbbé alakítás
    z_scores_np = z_scores.to_numpy()
    outliers_z = (z_scores_np > 3).flatten()  #max 3-at javasolnak
    # outlierek megtisztítása
    df = df[~outliers_z]

    return df