# FASE 4: Entrenament i avaluació dels models de regressió
# Compara 6 algorismes amb 5-fold CV sobre train i avaluació sobre val.
# El test set NO es toca en aquesta fase.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
K_FEATURES   = 19 # Features seleccionades per SelectKBest (veure exploració a preparacio_ml.py)

# 4.1. CÀRREGA DELS CONJUNTS IMPUTATS (SENSE seleccionar/escalar)
print("=" * 60)
print("1. CARREGA DELS CONJUNTS IMPUTATS")
print("=" * 60)
 
X_train = pd.read_csv("data/processed/X_train_imp.csv")
X_val   = pd.read_csv("data/processed/X_val_imp.csv")
 
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
 
print(f"\nTrain: {X_train.shape[0]} lots | Validation: {X_val.shape[0]} lots")
print(f"Features (abans de seleccionar): {X_train.shape[1]}")

# 4.2. DEFINICIÓ DELS PIEPLINES (imputer ja fet; selector+[scaler]+model)
print("\n" + "=" * 60)
print("2. ENTRENAMENT I AVALUACIO DELS MODELS (amb Pipeline)")
print("=" * 60)
 
def make_pipeline_scaled(model):
    """Selector + escalat + model. Per a models sensibles a l'escala."""
    return Pipeline([
        ("selector", SelectKBest(f_regression, k=K_FEATURES)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
 
def make_pipeline_raw(model):
    """Selector + model, sense escalat. Per a models basats en arbres."""
    return Pipeline([
        ("selector", SelectKBest(f_regression, k=K_FEATURES)),
        ("model", model),
    ])
    
# Models sensibles a l'escala: necessiten variables amb magnituds comparables
models_scaled = {
    "Linear Regression": make_pipeline_scaled(LinearRegression()),
    "Ridge": make_pipeline_scaled(Ridge(alpha=1.0)),
    "Lasso": make_pipeline_scaled(Lasso(alpha=0.1)),
    "SVR": make_pipeline_scaled(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
}
# Models basats en forest: no necessiten escalat (Random Forest, Gradient Boosting)
models_raw = {
    "Random Forest":     make_pipeline_raw(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    "Gradient Boosting": make_pipeline_raw(GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)),
}

# Calcula R², MAE i RMSE per a un model concret.
def evaluate(y_true, y_pred, name):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Model": name, "R²": round(r2, 4),
            "MAE": round(mae, 4), "RMSE": round(rmse, 4)}
 
# 5-fold cross-validation sobre train -- ara SENSE leakage:
# cada fold ajusta el seu propi scaler dins del Pipeline.
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
 
results = []
all_pipelines = {**models_scaled, **models_raw} # els 6 models junts
 
for name, pipe in all_pipelines.items():
     # Entrenem amb tot el train i avaluem sobre validation
    pipe.fit(X_train, y_train)
    y_pred_val = pipe.predict(X_val)
    metrics = evaluate(y_val, y_pred_val, name)

    # A més, fem 5-fold CV sobre train per veure com de robust és el model
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')
    metrics["CV R² mitjà"] = round(cv_scores.mean(), 4)
    metrics["CV R² std"]   = round(cv_scores.std(), 4)
 
    results.append(metrics)
    print(f"\n{name}:")
    print(f"  Val  --> R²={metrics['R²']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}")
    print(f"  5-CV --> R²={metrics['CV R² mitjà']:.4f} ± {metrics['CV R² std']:.4f}")
 
# 4.3. TAULA RESUM COMPARATIVA
print("\n" + "=" * 60)
print("3. TAULA RESUM (ordenada per R² Validation)")
print("=" * 60)
 
results_df = pd.DataFrame(results).set_index("Model")
results_df = results_df.sort_values("R²", ascending=False)
print(f"\n{results_df.to_string()}")
 
best_model_name = results_df["R²"].idxmax()
print(f"\nMillor model sobre Validation Set: {best_model_name}")
print(f"R²={results_df.loc[best_model_name, 'R²']:.4f} | "
      f"MAE={results_df.loc[best_model_name, 'MAE']:.4f} | "
      f"RMSE={results_df.loc[best_model_name, 'RMSE']:.4f}")

# 4.4. VISUALITZACIONS COMPARATIVES
print("\n" + "=" * 60)
print("4. VISUALITZACIONS")
print("=" * 60)
 
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = ['coral' if m == best_model_name else 'teal'
          for m in results_df.index]
 
axes[0].barh(results_df.index, results_df["R²"], color=colors, edgecolor='white')
axes[0].set_title("R² – Validation Set")
axes[0].set_xlabel("R²")
axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
 
axes[1].barh(results_df.index, results_df["MAE"], color=colors, edgecolor='white')
axes[1].set_title("MAE – Validation Set")
axes[1].set_xlabel("MAE (%)")
 
axes[2].barh(results_df.index, results_df["RMSE"], color=colors, edgecolor='white')
axes[2].set_title("RMSE – Validation Set")
axes[2].set_xlabel("RMSE (%)")
 
plt.suptitle("Comparació de models – Validation Set\n(el millor en color coral)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/04_comparativa_models.png", dpi=150)
plt.show()
plt.figure(figsize=(10, 5))

# CV R²: els models lineals (Regressió Lineal, Ridge, Lasso) tenen valors negatius per la inestabilitat numerica comentada abans.
# Es retallen visualment a -1.0
cv_means = results_df["CV R² mitjà"].clip(lower=-1.0)
cv_stds  = results_df["CV R² std"].clip(upper=1.0)
plt.barh(results_df.index, cv_means, xerr=cv_stds,
         color=colors, edgecolor='white', capsize=4)

# Anotem el valor real nomes per als models retallats
for i, (name, val) in enumerate(results_df["CV R² mitjà"].items()):
    if val < -1.0:
        plt.text(-0.98, i, f"  (real: {val:.0f})", va='center', fontsize=8, color='dimgray')

plt.title("R² mitjà – 5-Fold Cross-Validation sobre Train\n" "(barres d'error = desviació estàndard entre plecs; Pipeline sense leakage)")
plt.xlabel("R² mitjà CV")
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/05_cv_comparativa.png", dpi=150)
plt.show()

# 4.5. GUARDO EL MILLOR MODEL
# Guardo el millor model (per Validation R²) per a la fase d'optimització d'hiperparàmetres.
print("\n" + "=" * 60)
print("5. GUARDEM EL MILLOR MODEL")
print("=" * 60)
 
best_pipeline = all_pipelines[best_model_name]
joblib.dump(best_pipeline, "models/best_model_baseline.pkl")
 
print(f"\nModel guardat: models/best_model_baseline.pkl")
print(f"(Model: {best_model_name})")
