# FASE 3: Preparació del conjunt de dades per a Machine Learning
# Llegeix el dataset net (sortida de preprocessament.py) i el deixa llest
# per entrenar models: elimina variables amb data leakage, fa el split
# 80/10/10, imputa valors nuls i estandarditza.

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

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)
print("Imputació completada (fit únicament sobre train).")

# 4. SELECCIÓ DE FEATURES (SelectKBest, fitejada sobre train)
# ------------------------------------------------------------------------------
# SelectKBest selecciona les k variables amb F-score més alt (correlació
# lineal amb el Yield). Redueix la dimensionalitat de 64 a 15 features,
# evitant l'overfitting que genera tenir massa variables relatives a les
# mostres disponibles (152 lots).
print("\n" + "=" * 60)
print(f"4. SELECCIÓ DE FEATURES (SelectKBest, k={K_FEATURES})")
print("=" * 60)

selector = SelectKBest(f_regression, k=K_FEATURES)
X_train_sel = selector.fit_transform(X_train_imp, y_train)
X_val_sel   = selector.transform(X_val_imp)
X_test_sel  = selector.transform(X_test_imp)

feature_names    = X_train.columns.tolist()
selected_mask    = selector.get_support()
selected_features = [feature_names[i] for i, s in enumerate(selected_mask) if s]
f_scores = pd.Series(selector.scores_, index=feature_names)

print(f"\nFeatures seleccionades ({K_FEATURES}):")
for f in selected_features:
    print(f"  - {f}  (F={f_scores[f]:.2f})")

# 5. ESTANDARDITZACIÓ (fitejada sobre train)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. ESTANDARDITZACIÓ (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sel)
X_val_sc   = scaler.transform(X_val_sel)
X_test_sc  = scaler.transform(X_test_sel)

print(f"Mitjana train escalat (~0): {X_train_sc.mean():.4f}")
print(f"Std train escalat (~1):     {X_train_sc.std():.4f}")

# 6. GUARDEM ELS CONJUNTS I EL PIPELINE
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. GUARDEM CONJUNTS I PIPELINE")
print("=" * 60)

cols = selected_features
pd.DataFrame(X_train_sel, columns=cols).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_val_sel,   columns=cols).to_csv("data/processed/X_val.csv",   index=False)
pd.DataFrame(X_test_sel,  columns=cols).to_csv("data/processed/X_test.csv",  index=False)
pd.DataFrame(X_train_sc,  columns=cols).to_csv("data/processed/X_train_scaled.csv", index=False)
pd.DataFrame(X_val_sc,    columns=cols).to_csv("data/processed/X_val_scaled.csv",   index=False)
pd.DataFrame(X_test_sc,   columns=cols).to_csv("data/processed/X_test_scaled.csv",  index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv",     index=False)
y_test.to_csv("data/processed/y_test.csv",   index=False)

joblib.dump(imputer,  "models/imputer.pkl")
joblib.dump(selector, "models/selector.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
pd.Series(selected_features).to_csv("models/selected_features.csv",
                                     index=False, header=False)

print("Tots els fitxers guardats correctament.")
print(f"\nResum: {X_train.shape[0]} lots train | {K_FEATURES} features seleccionades")
