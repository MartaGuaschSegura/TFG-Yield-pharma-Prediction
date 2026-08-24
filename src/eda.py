# FASE 2: Anàlisi Exploratòria de Dades (EDA)
# Llegeix el dataset net generat per preprocessament.py i en fa l'exploració: distribució del Yield, outliers, duplicats i correlacions entre variables.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crea la carpeta de figures
os.makedirs("figures", exist_ok=True)

# 2.1. CÀRREGA DEL DATASET NET
# Aquest script parteix del fitxer ja netejat per preprocessament.py (dates convertides a durades, identificador OF separat, columnes ambexcés de nuls eliminades).

print("=" * 60)
print("1. CÀRREGA DEL DATASET NET")
print("=" * 60)

CSV_PATH = "/Users/martaguasch/Desktop/TFG/Netea i preprocessament/data/processed/dataset_net.csv"
TARGET_COL = "Yield (%)"

df = pd.read_csv(CSV_PATH)

print(f"\nDimensions: {df.shape[0]} lots x {df.shape[1]} variables")
print(f"Variable objectiu: {TARGET_COL}")

# 2.2. ANÀLISI EXPLORATÒRIA DE DADES (EDA)
print("\n" + "=" * 60)
print("2. ANÀLISI EXPLORATÒRIA DE DADES (EDA)")
print("=" * 60)

# Es genera una figura amb dos gràfics costat a costat (1 fila, 2 columnes): a l'esquerra un histograma per veure la forma de la distribució, i a la dreta un boxplot per identificar visualment els valors atípics (outliers).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma: cada barra mostra quants lots tenen un Yield dins d'aquell rang
axes[0].hist(df[TARGET_COL], bins=20, color='teal', edgecolor='white')

# Línia vermella discontínua marcant la mitjana
axes[0].axvline(df[TARGET_COL].mean(), color='red', linestyle='--',
                 label=f"Mitjana = {df[TARGET_COL].mean():.2f}%")
axes[0].set_title('Distribució del Yield (%)')
axes[0].set_xlabel('Yield (%)')
axes[0].set_ylabel('Freqüència')
axes[0].legend()

# Boxplot on es concentra el 50% central dels lots (IQR)
axes[1].boxplot(df[TARGET_COL], vert=True, patch_artist=True,
                 boxprops=dict(facecolor='teal', alpha=0.7))
axes[1].set_title('Boxplot del Yield (%)')
axes[1].set_ylabel('Yield (%)')

plt.tight_layout()
plt.savefig("figures/01_distribucio_yield.png", dpi=150)
plt.show()

# 2.3 Detecció d'outliers (mètode IQR)
# Identifica lots amb un Yield anormalment baix o alt respecte a la resta,que poden indicar incidents de procés a revisar.

Q1 = df[TARGET_COL].quantile(0.25)
Q3 = df[TARGET_COL].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

outliers = df[(df[TARGET_COL] < lim_inf) | (df[TARGET_COL] > lim_sup)]
print(f"\nLímits IQR: [{lim_inf:.2f}, {lim_sup:.2f}]")
print(f"Outliers detectats al Yield: {len(outliers)} lots")

if len(outliers) > 0:
    print(outliers[[TARGET_COL]].to_string())

# 2.3 Detecció de duplicats
n_dup = df.duplicated().sum()
print(f"\nFiles completament duplicades: {n_dup}")

# 2.4 Matriu de correlació
# S'identifica les variables de procés més correlacionades amb el Yield. Amb 65 variables numèriques, mostro només el top 15.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

top_corr = corr_matrix[TARGET_COL].abs().sort_values(ascending=False)
top_vars = top_corr.head(16).index.tolist()  # 15 variables + Yield

plt.figure(figsize=(12, 10))
sns.heatmap(df[top_vars].corr(), annot=True, fmt='.2f', cmap='viridis',
            center=0, square=True, linewidths=0.5, annot_kws={"size": 8})
plt.title('Matriu de Correlació – Top 15 variables + Yield')
plt.tight_layout()
plt.savefig("figures/02_correlacio.png", dpi=150)
plt.show()

print(f"\nTop 15 variables més correlacionades amb {TARGET_COL}:")
print(top_corr.iloc[1:16].to_string())  # [0] és Yield amb ell mateix (=1.0)

# 2.5 Gràfics de dispersió de les variables més rellevants
# Visualitzem la relació entre les 4 variables més correlacionades i el Yield, per detectar si la relació és lineal o no.
top4_vars = top_corr.iloc[1:5].index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, var in enumerate(top4_vars):
    axes[i].scatter(df[var], df[TARGET_COL], alpha=0.6, color='teal',
                     edgecolors='white')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel(TARGET_COL)
    axes[i].set_title(f'{var} vs Yield (r={corr_matrix.loc[var, TARGET_COL]:.2f})')

plt.tight_layout()
plt.savefig("figures/03_dispersio_top_variables.png", dpi=150)
plt.show()

# 2.6 Resum final de l'EDA
print("\n" + "=" * 60)
print("RESUM DE L'EDA")
print("=" * 60)
print(f"Lots analitzats: {df.shape[0]}")
print(f"Variables numèriques: {len(numeric_cols)}")
print(f"Yield mitjà: {df[TARGET_COL].mean():.2f}% (± {df[TARGET_COL].std():.2f})")
print(f"Outliers detectats: {len(outliers)}")
print(f"Duplicats detectats: {n_dup}")
print(f"\nFigures generades a la carpeta 'figures/':")
print("  1. 01_distribucio_yield.png")
print("  2. 02_correlacio.png")
print("  3. 03_dispersio_top_variables.png")
