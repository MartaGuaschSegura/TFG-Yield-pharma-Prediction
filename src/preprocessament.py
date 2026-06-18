#FASE ACTUAL: Càrrega del dataset + Neteja i preprocessament
# (les fases d'EDA, modelització, optimització i interpretabilitat s'aniran
#  afegint en propers commits)
import os 
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Crea les carpetes de sortida (codi net)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# Llavor aleatòria per reproducibilitat en tot el projecte
RANDOM_STATE = 42

# 1. CÀRREGA DEL DATASET
# ------------------------------------------------------------------------------
# El fitxer prové de l'extracció EBR del procés 5477-S03. Usa ';' com a
# separador de columnes; els decimals venen amb punt (no calen ajustos
# addicionals de 'decimal').

print("=" * 60)
print("1. CÀRREGA DEL DATASET")
print("=" * 60)

CSV_PATH = "/Users/martaguasch/Desktop/TFG/DATA EXTRACTION 5477-S03 (1).csv"
TARGET_COL = "Yield (%)" # objectiu
ID_COL = "OF" # Identificador únic de cada lot (Ordre de Fabricació)

df = pd.read_csv(CSV_PATH, sep=";")

print(f"\nDimensions del dataset: {df.shape[0]} lots x {df.shape[1]} variables")
print(f"Lots únics (columna {ID_COL}): {df[ID_COL].nunique()}")

print(f"\nTipus de variables:")
print(df.dtypes.value_counts())

print(f"\nValors nuls per columna (només columnes amb algun nul):")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0].to_string())

print(f"\nEstadística descriptiva de '{TARGET_COL}':")
print(df[TARGET_COL].describe())

# 3. NETEJA I PREPROCESSAMENT
# ------------------------------------------------------------------------------
# El dataset original té tres tipus de columnes a tractar de manera diferent:
#   a) Dates (Start/End de cada operació + DateTime Check Solution +
#      BATCH_CLOSURE_DATETIME) -> es transformen en durades numèriques (minuts).
#   b) Columnes numèriques de procés (pressions, temperatures, agitació,
#      càrregues en kg) -> es mantenen tal qual.
#   c) Identificador de lot (OF) -> es manté apart, no és una feature.

print("\n" + "=" * 60)
print("3. NETEJA I PREPROCESSAMENT")
print("=" * 60)

df_clean = df.copy()

# 3.1 Conversió de totes les columnes de data a datetime
# Deta pel nom: "Start", "End", "DateTime"
# o "CLOSURE". Format real del fitxer: dd/mm/yyyy H:MM (dayfirst=True).
date_keywords = ['Start', 'End', 'DateTime', 'CLOSURE']
date_cols = [c for c in df_clean.columns
             if any(k in c for k in date_keywords)]

print(f"\nColumnes de data detectades ({len(date_cols)}):")
for c in date_cols:
    print(f"  - {c}")

for col in date_cols:
    df_clean[col] = pd.to_datetime(df_clean[col], dayfirst=True, errors='coerce')

# Comprovem que la conversió no ha introduït nuls inesperats
nuls_post_conversio = df_clean[date_cols].isnull().sum()
print(f"\nNuls després de convertir a datetime (si n'hi ha, revisar format original):")
print(nuls_post_conversio[nuls_post_conversio > 0].to_string() or "  Cap nul nou introduït.")

# 3.2 Càlcul de durades (End - Start) en minuts
# Cobreix tant el patró estàndard "OP_N Start/End" com el cas especial
# "OP_ACN Loading Start/End", que no segueix la numeració OP_N.
start_cols = [c for c in date_cols if 'Start' in c]
n_durades = 0

for s_col in start_cols:
    e_col = s_col.replace('Start', 'End')
    if e_col in df_clean.columns:
        dur_col = s_col.replace('Start', 'Duration_min')
        df_clean[dur_col] = (df_clean[e_col] - df_clean[s_col]).dt.total_seconds() / 60
        n_durades += 1

print(f"\nVariables de durada creades: {n_durades}")
# Comprovació de durades negatives (indicarien error de registre a l'EBR,
# per exemple Start i End intercanviats, o operacions que travessen mitjanit
# mal registrades)
duration_cols = [c for c in df_clean.columns if 'Duration_min' in c]
durades_negatives = (df_clean[duration_cols] < 0).sum()
durades_negatives = durades_negatives[durades_negatives > 0]
if len(durades_negatives) > 0:
    print(f"\n[AVÍS] Durades negatives detectades (revisar lots concrets):")
    print(durades_negatives.to_string())
else:
    print("\nCap durada negativa detectada.")

# 3.3 Eliminem les columnes de data originals
# Un cop calculades les durades, les dates en format datetime no aporten
# valor predictiu directe als models de regressió (no són numèriques).
df_clean.drop(columns=date_cols, inplace=True)

# 3.4 Comprovació de columnes amb > 50% de valors nuls
# En aquest dataset concret pràcticament no n'hi ha (el procés EBR és
# consistent), però es manté la comprovació per robustesa davant futures
# extraccions de dades.
threshold = 0.5
null_ratio = df_clean.isnull().mean()
cols_to_drop = null_ratio[null_ratio > threshold].index.tolist()
print(f"\nColumnes eliminades per > 50% de nuls ({len(cols_to_drop)}): {cols_to_drop}")
df_clean.drop(columns=cols_to_drop, inplace=True)

# 3.5 Separem l'identificador de lot (OF)
# Es manté en un DataFrame apart per poder traçar resultats fins al lot
# original (útil per a la discussió de resultats), però NO s'utilitza
# com a feature d'entrada al model.
batch_ids = df_clean[ID_COL].copy()
df_clean.drop(columns=[ID_COL], inplace=True)

# 3.6 Eliminació de files sense valor de Yield 
# Un lot sense Yield registrat no es pot utilitzar ni per entrenar ni per avaluar.
files_abans = len(df_clean)
df_clean.dropna(subset=[TARGET_COL], inplace=True)
print(f"\nFiles eliminades per Yield nul: {files_abans - len(df_clean)}")

# 3.7 Comprovació final de tipus
# Totes les columnes restants haurien de ser numèriques (int o float).
non_numeric = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\n[AVÍS] Columnes no numèriques inesperades, revisar: {non_numeric}")
else:
    print("\nTotes les columnes restants són numèriques. ✓")

print(f"\nDataset net final: {df_clean.shape[0]} lots x {df_clean.shape[1]} variables")
print(f"(+ {len(batch_ids)} identificadors de lot guardats apart per a traçabilitat)")

# Guardem el dataset processat per a la següent fase (EDA / modelització)
df_clean.to_csv("/Users/martaguasch/Desktop/TFG/data/processed/dataset_net.csv", index=False)
#print("\n→ Dataset net guardat a: data/processed/dataset_net.csv")
