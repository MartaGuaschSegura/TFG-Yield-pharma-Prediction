# FASE 7: Interpretabilitat del model + Visualitzacions finals
# Analitza quines variables de procés tenen més impacte sobre el Yield
# mitjançant Feature Importance (Gini) i Permutation Importance.
# També genera les visualitzacions finals: predicció vs real i residus.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

RANDOM_STATE = 42

# 1. CÀRREGA DEL MODEL FINAL I DADES
print("=" * 60)
print("1. CÀRREGA DEL MODEL FINAL I DADES")
print("=" * 60)

# Carreguem el model final guardat a la fase d'optimització
model_final = joblib.load("models/model_final.pkl")
feature_names = pd.read_csv("models/selected_features.csv", header=None).squeeze().tolist()

print(f"Model final: {type(model_final).__name__}")
print(f"Features: {len(feature_names)}")

# Carreguem les dades (SVR usa escalades, RF usa sense escalar)
is_svr = isinstance(model_final, SVR)

if is_svr:
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    X_val   = pd.read_csv("data/processed/X_val_scaled.csv")
    X_test  = pd.read_csv("data/processed/X_test_scaled.csv")
    print("Usant dades escalades (model SVR)")
else:
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_val   = pd.read_csv("data/processed/X_val.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    print("Usant dades sense escalar (model Random Forest)")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()


# 2. FEATURE IMPORTANCE (GINI) — només si el model final és Random Forest (no es el cas, no s'executara)
# La importància de Gini mesura quant contribueix cada variable a reduir la
# impuresa dels nodes de l'arbre. Només disponible per a models basats en
# arbres (Random Forest, Gradient Boosting); no aplicable a SVR.

print("\n" + "=" * 60)
print("2. FEATURE IMPORTANCE")
print("=" * 60)

if not is_svr:
    importances = pd.Series(model_final.feature_importances_,
                            index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', color='teal',
                                    edgecolor='white')
    plt.title('Feature Importance (Gini) – Random Forest optimitzat')
    plt.xlabel('Importància')
    plt.tight_layout()
    plt.savefig("figures/07_feature_importance_gini.png", dpi=150)
    plt.show()
    print("→ Figura guardada: figures/07_feature_importance_gini.png")

    print("\nTop 10 variables per importància Gini:")
    print(importances.head(10).to_string())
else:
    print("Model SVR: Feature Importance de Gini no disponible.")
    print("S'utilitzarà únicament Permutation Importance.")


# 3. PERMUTATION IMPORTANCE
# La Permutation Importance mesura la caiguda de R² quan es permuta
# aleatòriament cada variable. És model-agnòstica (funciona amb SVR i RF)
# i menys sensible al biaix de cardinalitat que la importància de Gini.
# Es calcula sobre el validation set per evitar overfitting.

print("\n" + "=" * 60)
print("3. PERMUTATION IMPORTANCE (sobre Validation Set)")
print("=" * 60)

perm = permutation_importance(
    model_final, X_val, y_val,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': perm.importances_mean,
    'Std':        perm.importances_std,
}).sort_values('Importance', ascending=False)

print("\nTop 10 variables per Permutation Importance:")
print(perm_df[['Feature', 'Importance', 'Std']].head(10).to_string(index=False))

plt.figure(figsize=(10, 7))
plt.barh(
    perm_df['Feature'][::-1],
    perm_df['Importance'][::-1],
    xerr=perm_df['Std'][::-1],
    color='coral', edgecolor='white', capsize=3
)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('Permutation Importance – Validation Set\n'
          '(barres d\'error = std sobre 30 repeticions)')
plt.xlabel('Disminució de R² en permutar la variable')
plt.tight_layout()
plt.savefig("figures/08_permutation_importance.png", dpi=150)
plt.show()
print("→ Figura guardada: figures/08_permutation_importance.png")


# 4. VISUALITZACIONS FINALS: PREDICCIÓ VS REAL I RESIDUS

print("\n" + "=" * 60)
print("4. VISUALITZACIONS FINALS")
print("=" * 60)

y_pred_test = model_final.predict(X_test)

r2   = r2_score(y_test, y_pred_test)
mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nResultats finals sobre Test Set:")
print(f"  R²={r2:.4f} | MAE={mae:.4f}% | RMSE={rmse:.4f}%")

# Predicció vs Real
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

lim = [min(y_test.min(), y_pred_test.min()) - 0.5,
       max(y_test.max(), y_pred_test.max()) + 0.5]

axes[0].scatter(y_test, y_pred_test, alpha=0.7, color='teal',
                edgecolors='white', s=70)
axes[0].plot(lim, lim, 'r--', label='Predicció perfecta')
axes[0].set_xlabel('Yield real (%)')
axes[0].set_ylabel('Yield predit (%)')
axes[0].set_title(f'Predicció vs Real – Test Set\nR²={r2:.4f} | MAE={mae:.4f}%')
axes[0].legend()
axes[0].set_xlim(lim)
axes[0].set_ylim(lim)

# Residus
residuals = y_test.values - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.7, color='c',
                edgecolors='white', s=70)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Yield predit (%)')
axes[1].set_ylabel('Residu (real − predit)')
axes[1].set_title('Anàlisi de Residus – Test Set')

plt.tight_layout()
plt.savefig("figures/09_prediccio_vs_real.png", dpi=150)
plt.show()
print("→ Figura guardada: figures/09_prediccio_vs_real.png")

# Distribució dels residus
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=10, color='teal', edgecolor='white', alpha=0.8)
plt.axvline(0, color='black', linestyle='--')
plt.axvline(residuals.mean(), color='red', linestyle='-',
            label=f'Mitjana = {residuals.mean():.3f}%')
plt.title('Distribució dels Residus – Test Set')
plt.xlabel('Residu (%)')
plt.ylabel('Freqüència')
plt.legend()
plt.tight_layout()
plt.savefig("figures/10_distribucio_residus.png", dpi=150)
plt.show()
print("→ Figura guardada: figures/10_distribucio_residus.png")


# 5. RESUM FINAL

print("\n" + "=" * 60)
print("RESUM FINAL DEL PROJECTE")
print("=" * 60)
print(f"\nModel final: {type(model_final).__name__} optimitzat")
print(f"Features usades ({len(feature_names)}):")
for f in feature_names:
    print(f"  - {f}")
print(f"\nResultats sobre Test Set:")
print(f"  R²   = {r2:.4f}")
print(f"  MAE  = {mae:.4f}%")
print(f"  RMSE = {rmse:.4f}%")
print(f"\nFigures generades:")
figs = ["07_feature_importance_gini.png (si RF)",
        "08_permutation_importance.png",
        "09_prediccio_vs_real.png",
        "10_distribucio_residus.png"]
for f in figs:
    print(f"  - {f}")
