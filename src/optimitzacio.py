# FASE 5: Optimització d'hiperparàmetres
# S'optimitzen els dos millors models de la fase anterior (SVR i Random
# Forest) amb RandomizedSearchCV i validació creuada de 5 plecs sobre train.
# El test set segueix sense tocar-se.

RANDOM_STATE = 42
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform

# 5.1. CÀRREGA DELS CONJUNTS
# ----------------------------------------------------------------------------------
print("=" * 60)
print("1. CÀRREGA DELS CONJUNTS")
print("=" * 60)

# Random Forest --> dades sense escalar
X_train = pd.read_csv("data/processed/X_train.csv")
X_val   = pd.read_csv("data/processed/X_val.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")

# SVR --> dades escalades
X_train_sc = pd.read_csv("data/processed/X_train_scaled.csv")
X_val_sc   = pd.read_csv("data/processed/X_val_scaled.csv")
X_test_sc  = pd.read_csv("data/processed/X_test_scaled.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"Train: {X_train.shape[0]} lots | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def evaluate(y_true, y_pred, nom):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {nom}: R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")
    return r2, mae, rmse

# 5.2. OPTIMITZACIÓ SVR
# ---------------------------------------------------------------------------------
# SVR és sensible als hiperparàmetres C (penalització), epsilon (marge de
# tolerància) i gamma (amplada del kernel RBF). Explorem un rang ampli per
# trobar la combinació que millora el R² sobre el validation set.

print("\n" + "=" * 60)
print("2. OPTIMITZACIÓ SVR")
print("=" * 60)

param_svr = {
    'C': uniform(0.01, 100), # rang: 0.01 --> 100
    'epsilon': uniform(0.01, 2.0), # rang: 0.01 --> 2.01
    'gamma':['scale', 'auto'] + list(uniform(0.001, 0.5).rvs(5, random_state=RANDOM_STATE)),
    'kernel':['rbf', 'linear'],
}

search_svr = RandomizedSearchCV(
    SVR(),
    param_distributions=param_svr,
    n_iter=100,
    cv=cv5,
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
search_svr.fit(X_train_sc, y_train)

print(f"\nMillors hiperparàmetres SVR:")
for k, v in search_svr.best_params_.items():
    print(f"{k}: {v}")
print(f"Millor R² CV (train): {search_svr.best_score_:.4f}")

best_svr = search_svr.best_estimator_
print(f"\nAvaluació SVR optimitzat:")
evaluate(y_val, best_svr.predict(X_val_sc), "Validation")

# 5.3. OPTIMITZACIÓ RANDOM FOREST
# ---------------------------------------------------------------------------------
# Random Forest té múltiples hiperparàmetres que controlen la complexitat
# de cada arbre i de l'ensemble. Amb 120 mostres, cal limitar la profunditat
# per evitar overfitting.

print("\n" + "=" * 60)
print("3. OPTIMITZACIÓ RANDOM FOREST")
print("=" * 60)

param_rf = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 3, 5, 7, 10, 15],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
    'bootstrap': [True, False],
}

search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=param_rf,
    n_iter=100,
    cv=cv5,
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0,
)
search_rf.fit(X_train, y_train)

print(f"\nMillors hiperparàmetres Random Forest:")
for k, v in search_rf.best_params_.items():
    print(f"  {k}: {v}")
print(f"Millor R² CV (train): {search_rf.best_score_:.4f}")

best_rf = search_rf.best_estimator_
print(f"\nAvaluació RF optimitzat:")
evaluate(y_val, best_rf.predict(X_val), "Validation")


# 5.4. COMPARACIÓ BASELINE VS OPTIMITZAT
# --------------------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. COMPARACIÓ BASELINE VS OPTIMITZAT")
print("=" * 60)

svr_base = SVR(kernel='rbf', C=1.0, epsilon=0.1).fit(X_train_sc, y_train)
rf_base = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)

r2_svr_base,  mae_svr_base,  rmse_svr_base  = evaluate(y_val, svr_base.predict(X_val_sc), "SVR baseline    ")
r2_svr_opt,   mae_svr_opt,   rmse_svr_opt   = evaluate(y_val, best_svr.predict(X_val_sc), "SVR optimitzat  ")
r2_rf_base,   mae_rf_base,   rmse_rf_base   = evaluate(y_val, rf_base.predict(X_val),     "RF baseline     ")
r2_rf_opt,    mae_rf_opt,    rmse_rf_opt    = evaluate(y_val, best_rf.predict(X_val),      "RF optimitzat   ")

comp_df = pd.DataFrame({
    "Model":  ["SVR baseline", "SVR optimitzat", "RF baseline", "RF optimitzat"],
    "R²":     [r2_svr_base,  r2_svr_opt,  r2_rf_base,  r2_rf_opt],
    "MAE":    [mae_svr_base, mae_svr_opt, mae_rf_base, mae_rf_opt],
    "RMSE":   [rmse_svr_base,rmse_svr_opt,rmse_rf_base,rmse_rf_opt],
}).set_index("Model").round(4)

print(f"\n{comp_df.to_string()}")

# Determina el millor model global
best_overall_name = comp_df["R²"].idxmax()
print(f"\nMillor model global: {best_overall_name}")

# 5.5. VISUALITZACIÓ COMPARATIVA
# ---------------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ['coral' if m == best_overall_name else 'teal'
          for m in comp_df.index]

for ax, metric in zip(axes, ["R²", "MAE", "RMSE"]):
    ax.barh(comp_df.index, comp_df[metric], color=colors, edgecolor='white')
    ax.set_title(f"{metric} – Validation Set")
    ax.set_xlabel(metric)
    if metric == "R²":
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

plt.suptitle("Baseline vs Optimitzat – SVR i Random Forest\n(millor en verd)",
             fontsize=12)
plt.tight_layout()
plt.savefig("figures/06_optimitzacio_comparativa.png", dpi=150)
plt.show()
print("\n→ Figura guardada: figures/06_optimitzacio_comparativa.png")


# 5.6. AVALUACIÓ FINAL SOBRE EL TEST SET
# ---------------------------------------------------------------------------------------
# el test set es toca UNA SOLA VEGADA, aquí, amb el model
# definitiu ja escollit. No es pot tornar a usar per prendre decisions.

print("\n" + "=" * 60)
print("6. AVALUACIÓ FINAL SOBRE EL TEST SET")
print("=" * 60)

# Seleccionem el millor model entre SVR i RF optimitzats
if r2_svr_opt >= r2_rf_opt:
    best_final = best_svr
    X_test_final = X_test_sc
    nom_final = "SVR optimitzat"
else:
    best_final = best_rf
    X_test_final = X_test
    nom_final = "Random Forest optimitzat"

y_pred_test = best_final.predict(X_test_final)
r2_test, mae_test, rmse_test = evaluate(y_test, y_pred_test,
                                         f"{nom_final} – TEST SET FINAL")

print(f"\n RESULTAT DEFINITIU (test set, no tocat fins ara)")
print(f"Model: {nom_final}")
print(f"R²={r2_test:.4f} | MAE={mae_test:.4f}% | RMSE={rmse_test:.4f}%")

# Guardem el model final
joblib.dump(best_final, "models/model_final.pkl")
print(f"\nModel final guardat: models/model_final.pkl")
