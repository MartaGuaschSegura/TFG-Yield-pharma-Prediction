# FASE 3: Preparació del conjunt de dades per a Machine Learning
# Llegeix el dataset net (sortida de preprocessament.py) i el deixa llest
# per entrenar models: elimina variables amb data leakage, fa el split
# 80/10/10, imputa valors nuls.

import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42
K_FEATURES   = 15   # Features seleccionades per SelectKBest

# 1. CÀRREGA DEL DATASET NET
#----------------------------------------------------------------------------
print("=" * 60)
print("1. CÀRREGA DEL DATASET NET")
print("=" * 60)

TARGET_COL   = "Yield (%)"
LEAKAGE_COLS = ["Obtained quantity (kg)"]

df = pd.read_csv("/Users/martaguasch/Desktop/TFG/Netea i preprocessament/data/processed/dataset_net.csv")
print(f"Dimensions originals: {df.shape[0]} lots x {df.shape[1]} variables")

cols_drop = [c for c in LEAKAGE_COLS if c in df.columns]
df_ml = df.drop(columns=cols_drop)
print(f"Variables eliminades per data leakage: {cols_drop}")

X = df_ml.drop(columns=[TARGET_COL])
y = df_ml[TARGET_COL]

# 2. THREE-WAY SPLIT (80/10/10)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. THREE-WAY SPLIT (80 / 10 / 10)")
print("=" * 60)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=1/9, random_state=RANDOM_STATE)

print(f"Train:      {X_train.shape[0]} lots ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]} lots ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:       {X_test.shape[0]} lots ({X_test.shape[0]/len(X)*100:.1f}%)")

# 3. IMPUTACIÓ (mediana, fitejada sobre train)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. IMPUTACIÓ DE VALORS NULS (mediana)")
print("=" * 60)

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imp   = pd.DataFrame(imputer.transform(X_val),   columns=X_val.columns)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=X_test.columns)
print("Imputació completada (fit únicament sobre train).")

# 4. GUARDEM ELS CONJUNTS IMPUTATS (sense seleccionar ni escalar)
# ----------------------------------------------------------------------------
#    Aquests son els fitxers que faran servir entrenament_models.py
#    i optimitzacio.py

print("\n" + "=" * 60)
print("4. GUARDEM CONJUNTS IMPUTATS (65 features, sense seleccionar)")
print("=" * 60)
 
X_train_imp.to_csv("data/processed/X_train_imp.csv", index=False)
X_val_imp.to_csv("data/processed/X_val_imp.csv", index=False)
X_test_imp.to_csv("data/processed/X_test_imp.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
 
joblib.dump(imputer, "models/imputer.pkl")

# 5. SELECCIÓ DE FEATURES I ESCALAT (SelectKBest, fitejada sobre train)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"5. SELECCIO DE FEATURES INFORMATIVA (SelectKBest, k={K_FEATURES})")
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
 
scaler_info = StandardScaler()
scaler_info.fit(selector_info.transform(X_train_imp))
 
joblib.dump(selector_info, "models/selector.pkl")
joblib.dump(scaler_info, "models/scaler.pkl")
pd.Series(selected_features).to_csv(
    "models/selected_features.csv", index=False, header=False)
 
print("\nGuardats (informatiu / model final): selector.pkl, scaler.pkl, selected_features.csv")
print("La CV de les fases 4 i 5 fa servir el seu PROPI selector/scaler")
print("dins d'un Pipeline, ajustat per separat a cada fold.")
 
print(f"\nResum: {X_train.shape[0]} lots train | {K_FEATURES} features seleccionades (informatiu)")
 
