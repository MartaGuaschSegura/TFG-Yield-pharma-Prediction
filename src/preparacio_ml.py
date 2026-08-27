# FASE 3: Preparació del conjunt de dades per a Machine Learning
# Llegeix el dataset net (sortida de preprocessament.py) i el deixa llest per entrenar models: elimina variables amb data leakage, fa el split 80/10/10, imputa valors nuls.

import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

os.makedirs("models", exist_ok=True)
RANDOM_STATE = 42

# 1. CÀRREGA DEL DATASET NET
print("=" * 60)
print("1. CÀRREGA DEL DATASET NET")
print("=" * 60)

TARGET_COL   = "Yield (%)"
LEAKAGE_COLS = ["Obtained quantity (kg)"]

df = pd.read_csv("data/processed/dataset_net.csv")
print(f"Dimensions originals: {df.shape[0]} lots x {df.shape[1]} variables")

# S'eliminala variable "Obtained quantity (kg)"
df_without_leakage = df.drop(columns=LEAKAGE_COLS)
print(f"Variables eliminades per data leakage: {LEAKAGE_COLS}")

# Es separa les dades en X (variables predictores, tot excepte el Yield) i y (la variable objectiu que es vol predir: el Yield)
X = df_without_leakage.drop(columns=[TARGET_COL])
y = df_without_leakage[TARGET_COL]

# 2. THREE-WAY SPLIT (80/10/10)
print("\n" + "=" * 60)
print("2. THREE-WAY SPLIT (80 / 10 / 10)")
print("=" * 60)

# Dividim les dades en 3 conjunts per poder fer optimització d'hiperparàmetres sense contaminar mai el conjunt de test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=1/9, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape[0]} lots ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} lots ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test: {X_test.shape[0]} lots ({X_test.shape[0]/len(X)*100:.1f}%)")

# La mediana és més robusta que la mitjana davant valors atípics: com que el dataset té 2 outliers detectats a l'EDA, la mitjana quedaria esbiaixada cap a valors alts
imputer = SimpleImputer(strategy="median")

# Calculo la mediana de cada columna mirant només el train, i de seguida substitueix els seus propis valors buits.
# Es reconverteix a DataFrame perquè fit_transform retorna un array de numpy sense noms de columna.
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# transform (sense fit) sobre val i test: apliquem les mateixes medianes ja calculades amb train, sense tornar-les a calcular. Així evitem que informació de validation/test es coli cap al procés d'entrenament.
X_val_imp   = pd.DataFrame(imputer.transform(X_val),   columns=X_val.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=X_test.columns)

# 3. GUARDEM ELS CONJUNTS IMPUTATS (sense seleccionar ni escalar)
#    Aquests son els fitxers que faran servir entrenament_models.py i optimitzacio.py
# (65 features, sense seleccionar)

X_train_imp.to_csv("data/processed/X_train_imp.csv", index=False)
X_val_imp.to_csv("data/processed/X_val_imp.csv", index=False)
X_test_imp.to_csv("data/processed/X_test_imp.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
 
joblib.dump(imputer, "models/imputer.pkl")

# 4. EXPLORACIÓ DEL VALOR DE K
# Provem diferents valors de k i comparem el R2 mitjà en validació creuada de 5 plecs, per justificar quin nombre de features triem. El Pipeline garanteix que el SelectKBest s'ajusta per separat a cada fold (sense leakage).
print("\n" + "=" * 60)
print("3. EXPLORACIÓ DEL VALOR DE K")
print("=" * 60)
 
cv_exploracio = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
valors_k = range(10, 31)
resultats_k = []
 
for k in valors_k:
    pipeline_k = Pipeline([
        ("selector", SelectKBest(f_regression, k=k)),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])
    scores = cross_val_score(pipeline_k, X_train_imp, y_train, cv=cv_exploracio, scoring="r2")
    resultats_k.append({"k": k, "R2_mitja": scores.mean(), "R2_std": scores.std()})
    print(f"k={k:2d}  ->  R2 mitja CV = {scores.mean():.4f} (+/- {scores.std():.4f})")
 
resultats_k_df = pd.DataFrame(resultats_k)
millor_k = int(resultats_k_df.loc[resultats_k_df["R2_mitja"].idxmax(), "k"])
print(f"\nMillor k trobat segons R2 mitja CV: k={millor_k}")
 
K_FEATURES = millor_k

# 5. SELECCIÓ DE FEATURES I ESCALAT (SelectKBest, fitejada sobre train)
# Aquest bloc NOMÉS serveix per veure per pantalla i documentar a la memòria quines són les 19 variables més rellevants quan es fa servir tot el train disponible. NO és el selector que fa servir realment el  model: aquell està dins del Pipeline de entrenament_models.py i optimitzacio.py, ajustat per separat a cada fold de la validació creuada per evitar data leakage.
print("\n" + "=" * 60)
print(f"4. SELECCIO DE FEATURES INFORMATIVA (SelectKBest, k={K_FEATURES})")
print("=" * 60)
 
selector_info = SelectKBest(f_regression, k=K_FEATURES)
selector_info.fit(X_train_imp, y_train)
 
feature_names = X_train_imp.columns.tolist()
selected_mask = selector_info.get_support()
selected_features = [feature_names[i] for i, s in enumerate(selected_mask) if s]
f_scores = pd.Series(selector_info.scores_, index=feature_names)
 
print(f"\nFeatures seleccionades sobre TOT el train ({K_FEATURES}):")
for f in selected_features:
    print(f"  - {f}  (F={f_scores[f]:.2f})")
