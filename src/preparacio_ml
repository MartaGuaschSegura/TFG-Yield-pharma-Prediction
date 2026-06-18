# FASE 3: Preparació del conjunt de dades per a Machine Learning
# Llegeix el dataset net (sortida de preprocessament.py) i el deixa llest
# per entrenar models: elimina variables amb data leakage, fa el split
# 80/10/10, imputa valors nuls i estandarditza.
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Crea les carpetes necessàries si no existeixen
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42

# 1. CÀRREGA DEL DATASET NET
#----------------------------------------------------------------------------
print("=" * 60)
print("1. CÀRREGA DEL DATASET NET")
print("=" * 60)

CSV_PATH = "/Users/martaguasch/Desktop/TFG/Netea i preprocessament/data/processed/dataset_net.csv"
TARGET_COL = "Yield (%)"

df = pd.read_csv(CSV_PATH)
print(f"\nDimensions: {df.shape[0]} lots x {df.shape[1]} variables")

# 2. ELIMINACIÓ DE VARIABLES AMB DATA LEAKAGE
#---------------------------------------------------------------------------
# Amb l'EDA vaig detectar que 'Obtained quantity (kg)' té una correlació molt alta
# (r=0.76) amb el Yield perquè és gairebé el numerador del propi càlcul del
# rendiment (Yield = Obtained quantity / Quantity esperada x 100).
# Si la deixéssim com a feature, el model "trampejaria": aprendria a predir
# el Yield mirant gairebé el seu propi resultat, en lloc d'aprendre dels
# paràmetres reals de procés. Per això l'elimino.

print("\n" + "=" * 60)
print("2. ELIMINACIÓ DE VARIABLES AMB DATA LEAKAGE")
print("=" * 60)

LEAKAGE_COLS = ["Obtained quantity (kg)"]

cols_presents = [c for c in LEAKAGE_COLS if c in df.columns]
print(f"\nVariables eliminades per data leakage: {cols_presents}")
df_ml = df.drop(columns=cols_presents)

print(f"Dimensions després d'eliminar leakage: {df_ml.shape[0]} lots x {df_ml.shape[1]} variables")

# 3. SEPARACIÓ DE FEATURES (X) I VARIABLE OBJECTIU (y)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. SEPARACIÓ DE FEATURES (X) I VARIABLE OBJECTIU (y)")
print("=" * 60)

X = df_ml.drop(columns=[TARGET_COL])
y = df_ml[TARGET_COL]

feature_names = X.columns.tolist()
print(f"\nNombre de features: {len(feature_names)}")
print(f"Variable objectiu: {TARGET_COL}")
print(f"  min={y.min():.2f} | max={y.max():.2f} | mitjana={y.mean():.2f} | std={y.std():.2f}")

# 4. THREE-WAY SPLIT: TRAIN (80%) / VALIDATION (10%) / TEST (10%)
# -----------------------------------------------------------------------------
# Utilitzo un three-way split en lloc del clàssic 80/20 per evitar
# contaminar el conjunt de test amb decisions de modelització (ajust
# d'hiperparàmetres, selecció d'algorisme). El validation set actua de
# "proxy test" durant tot el desenvolupament; el test només es toca una
# vegada, al final, amb el model ja definitiu.

print("\n" + "=" * 60)
print("4. THREE-WAY SPLIT (80/10/10)")
print("=" * 60)

# 1r split: separa train+val (90%) del test final (10%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=RANDOM_STATE
)

# 2n split: separa train (80% del total) del validation (10% del total)
# 1/9 de 90% = 10% del total original
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=1/9, random_state=RANDOM_STATE
)

print(f"\nMides dels conjunts:")
print(f"  Train:      {X_train.shape[0]} lots ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation: {X_val.shape[0]} lots ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:       {X_test.shape[0]} lots ({X_test.shape[0]/len(X)*100:.1f}%)")

# 5. IMPUTACIÓ DE VALORS NULS
# --------------------------------------------------------------------------------
# S'usa la mediana (més robusta que la mitjana davant outliers en dades de
# procés industrial). MOLT IMPORTANT: l'imputer es fiteja NOMÉS sobre train,
# i després només es transformen val i test, mai es refiteja sobre ells
# (evita data leakage entre conjunts).

print("\n" + "=" * 60)
print("5. IMPUTACIÓ DE VALORS NULS")
print("=" * 60)

n_nuls_train = X_train.isnull().sum().sum()
print(f"\nValors nuls a train abans d'imputar: {n_nuls_train}")

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)   # fit + transform sobre train
X_val_imp   = imputer.transform(X_val)          # només transform
X_test_imp  = imputer.transform(X_test)         # només transform

print("Imputació completada (mediana, fitejada únicament sobre train).")

# 6. ESTANDARDITZACIÓ
# ----------------------------------------------------------------------------------
# StandardScaler centra les variables a mitjana 0 i variància 1. Necessari
# per a models sensibles a l'escala (Regressió Lineal, Ridge, Lasso, SVR).
# Random Forest i Gradient Boosting no ho requereixen, però mantenir les
# dades estandarditzades no els afecta negativament.
# Igual que l'imputer, el scaler es fiteja NOMÉS sobre train.

print("\n" + "=" * 60)
print("6. ESTANDARDITZACIÓ")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)  # fit + transform sobre train
X_val_sc   = scaler.transform(X_val_imp)         # només transform
X_test_sc  = scaler.transform(X_test_imp)        # només transform

print("\nEstandardització completada (fitejada únicament sobre train).")
print(f"Mitjana de train després d'escalar (~0 esperat): {X_train_sc.mean():.4f}")
print(f"Std de train després d'escalar (~1 esperat): {X_train_sc.std():.4f}")

# 7. GUARDEM ELS CONJUNTS PROCESSATS
# ----------------------------------------------------------------------------------
# Es guarden tant les versions imputades (sense escalar, per a Random Forest
# i Gradient Boosting) com les versions escalades (per a Ridge, Lasso, SVR,
# Regressió Lineal), per poder-les reutilitzar directament a la fase
# d'entrenament de models sense repetir aquest pipeline.

print("\n" + "=" * 60)
print("7. GUARDEM ELS CONJUNTS PROCESSATS")
print("=" * 60)

# Reconstrueixo DataFrames amb els noms de columnes originals per llegibilitat
X_train_imp_df = pd.DataFrame(X_train_imp, columns=feature_names)
X_val_imp_df   = pd.DataFrame(X_val_imp, columns=feature_names)
X_test_imp_df  = pd.DataFrame(X_test_imp, columns=feature_names)

X_train_sc_df = pd.DataFrame(X_train_sc, columns=feature_names)
X_val_sc_df   = pd.DataFrame(X_val_sc, columns=feature_names)
X_test_sc_df  = pd.DataFrame(X_test_sc, columns=feature_names)

# Dades sense escalar (per a Random Forest / Gradient Boosting)
X_train_imp_df.to_csv("data/processed/X_train.csv", index=False)
X_val_imp_df.to_csv("data/processed/X_val.csv", index=False)
X_test_imp_df.to_csv("data/processed/X_test.csv", index=False)

# Dades escalades (per a Ridge / Lasso / SVR / Regressió Lineal)
X_train_sc_df.to_csv("data/processed/X_train_scaled.csv", index=False)
X_val_sc_df.to_csv("data/processed/X_val_scaled.csv", index=False)
X_test_sc_df.to_csv("data/processed/X_test_scaled.csv", index=False)

# Variable objectiu (igual per a tots els models)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_val.to_csv("data/processed/y_val.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# Guardo també l'imputer i el scaler ja fitejats, per si calen més endavant
# (per exemple per fer prediccions sobre lots nous)
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nFitxers guardats a data/processed/:")
for f in ["X_train.csv", "X_val.csv", "X_test.csv",
          "X_train_scaled.csv", "X_val_scaled.csv", "X_test_scaled.csv",
          "y_train.csv", "y_val.csv", "y_test.csv"]:
    print(f"  - {f}")
print("\nImputer i scaler guardats a models/ (imputer.pkl, scaler.pkl)")

print("\n" + "=" * 60)
print("PREPARACIÓ PER A ML COMPLETADA")
print("=" * 60)
print(f"Dataset llest amb {len(feature_names)} features per entrenar models.")
