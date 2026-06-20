# FASE 4: Entrenament i avaluació dels models de regressió
# Compara 6 algorismes amb 5-fold CV sobre train i avaluació sobre val.
# El test set NO es toca en aquesta fase.

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# 4.1. CÀRREGA DELS CONJUNTS PREPROCESSATS
# ----------------------------------------------------------------------------------

print("=" * 60)
print("1. CÀRREGA DELS CONJUNTS PREPROCESSATS")
print("=" * 60)

# Dades sense escalar -> per a Random Forest i Gradient Boosting
X_train = pd.read_csv("data/processed/X_train.csv")
X_val = pd.read_csv("data/processed/X_val.csv")

# Dades escalades -> per a Linear Regression, Ridge, Lasso, SVR
X_train_sc = pd.read_csv("data/processed/X_train_scaled.csv")
X_val_sc = pd.read_csv("data/processed/X_val_scaled.csv")

# Variable objectiu
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_val = pd.read_csv("data/processed/y_val.csv").squeeze()

print(f"\nTrain: {X_train.shape[0]} lots | Validation: {X_val.shape[0]} lots")
print(f"Features: {X_train.shape[1]}")


# 4.2. DEFINICIÓ DELS MODELS I FUNCIÓ D'AVALUACIÓ
# ----------------------------------------------------------------------------------

print("\n" + "=" * 60)
print("2. ENTRENAMENT I AVALUACIÓ DELS MODELS")
print("=" * 60)

# Models que necessiten dades escalades
models_scaled = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

# Models que NO necessiten escalat
models_raw = {"Random Forest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
             }

# Funció per calcular R², MAE i RMSE
def evaluate(y_true, y_pred, name):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Model": name, "R²": round(r2, 4),
            "MAE": round(mae, 4), "RMSE": round(rmse, 4)}

# 5-fold cross-validation sobre train
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []

# Models escalats
for name, model in models_scaled.items():
    model.fit(X_train_sc, y_train)
    y_pred_val = model.predict(X_val_sc)
    metrics = evaluate(y_val, y_pred_val, name)

    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='r2')
    metrics["CV R² mitjà"] = round(cv_scores.mean(), 4)
    metrics["CV R² std"]   = round(cv_scores.std(), 4)

    results.append(metrics)
    print(f"\n{name}:")
    print(f"  Val  --> R²={metrics['R²']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}")
    print(f"  5-CV --> R²={metrics['CV R² mitjà']:.4f} ± {metrics['CV R² std']:.4f}")

# Models sense escalar
for name, model in models_raw.items():
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    metrics = evaluate(y_val, y_pred_val, name)

    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    metrics["CV R² mitjà"] = round(cv_scores.mean(), 4)
    metrics["CV R² std"]   = round(cv_scores.std(), 4)

    results.append(metrics)
    print(f"\n{name}:")
    print(f"  Val --> R²={metrics['R²']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}")
    print(f"  5-CV --> R²={metrics['CV R² mitjà']:.4f} ± {metrics['CV R² std']:.4f}")

# 4.3. TAULA RESUM COMPARATIVA
# ----------------------------------------------------------------------------------

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
# ----------------------------------------------------------------------------------

print("\n" + "=" * 60)
print("4. VISUALITZACIONS")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = ['coral' if m == best_model_name else 'teal'
          for m in results_df.index]

# R² Validation
axes[0].barh(results_df.index, results_df["R²"],
             color=colors, edgecolor='white')
axes[0].set_title("R² – Validation Set")
axes[0].set_xlabel("R²")
axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)

# MAE Validation
axes[1].barh(results_df.index, results_df["MAE"],
             color=colors, edgecolor='white')
axes[1].set_title("MAE – Validation Set")
axes[1].set_xlabel("MAE (%)")

# RMSE Validation
axes[2].barh(results_df.index, results_df["RMSE"],
             color=colors, edgecolor='white')
axes[2].set_title("RMSE – Validation Set")
axes[2].set_xlabel("RMSE (%)")

plt.suptitle("Comparació de models – Validation Set\n"
             "(el millor en verd)", fontsize=12)
plt.tight_layout()
plt.savefig("figures/04_comparativa_models.png", dpi=150)
plt.show()
print("Figura guardada: figures/04_comparativa_models.png")

# CV R² amb barres d'error (std)
plt.figure(figsize=(10, 5))
plt.barh(results_df.index,
         results_df["CV R² mitjà"],
         xerr=results_df["CV R² std"],
         color=colors, edgecolor='white', capsize=4)
plt.title("R² mitjà – 5-Fold Cross-Validation sobre Train\n"
          "(barres d'error = desviació estàndard entre plecs)")
plt.xlabel("R² mitjà CV")
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/05_cv_comparativa.png", dpi=150)
plt.show()
print("Figura guardada: figures/05_cv_comparativa.png")

# 4.5. GUARDO EL MILLOR MODEL
# ----------------------------------------------------------------------------------
# Guardo el millor model (per Validation R²) per a la fase d'optimització
# d'hiperparàmetres.

print("\n" + "=" * 60)
print("5. GUARDEM EL MILLOR MODEL")
print("=" * 60)

# Recuperem el model entrenat (el que hem fitejat abans)
all_models = {**models_scaled, **models_raw}
best_model = all_models[best_model_name]
joblib.dump(best_model, "models/best_model_baseline.pkl")

print(f"\nModel guardat: models/best_model_baseline.pkl")
print(f"(Model: {best_model_name})")
