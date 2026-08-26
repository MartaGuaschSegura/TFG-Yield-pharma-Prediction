# TFG – Predicció del Yield en Producció Farmacèutica (Procés 5477-S03)

Treball de Fi de Grau – Enginyeria Biomèdica – Universitat Rovira i Virgili (URV)

## Objectiu

Construir un model de Machine Learning capaç de predir el rendiment (Yield, %) d'un lot de producció farmacèutica a partir dels paràmetres de procés registrats (temperatures, pressions, agitació, càrregues d'ACN, etc.) al llarg de fins a 20 operacions seqüencials del procés 5477-S03.

Les dades provenen de l'Electronic Batch Record (EBR) implementat a l'empresa.

## Estat actual del projecte

- Càrrega del dataset
- Neteja i preprocessament (conversió de dates a durades, separació de l'identificador de lot)
- Anàlisi exploratòria de dades (EDA)
- Preparació per a Machine Learning (split 80/10/10, imputació de valors nuls)
- Entrenament i avaluació de models de regressió
- Optimització d'hiperparàmetres
- Interpretabilitat del model (Permutation Importance)
- Visualitzacions finals

## Estructura del repositori

```
tfg-yield-prediction/
├── data/
│   ├── raw/          ← CSV original de l'EBR (NO es puja, veure .gitignore)
│   └── processed/     ← Dades netes generades pel codi (NO es puja)
├── src/
│   ├── preprocessament.py      ← Càrrega + neteja + preprocessament
│   ├── eda.py                  ← Anàlisi exploratòria
│   ├── preparacio_ml.py        ← Split, imputació i exploració del valor de k
│   ├── entrenament_models.py   ← Entrenament i avaluació de 6 models de regressió amb 5-fold CV
│   ├── optimitzacio.py         ← Optimització d'hiperparàmetres SVR i RF amb RandomizedSearchCV
│   └── interpretabilitat.py    ← Interpretabilitat i visualitzacions finals
├── figures/           ← Figures generades automàticament pel codi
├── .gitignore
├── requirements.txt
└── README.md
```
## Com executar el pipeline?

Els scripts s'han d'executar en aquest ordre, des de la carpeta arrel del repositori:
```
install requirements.txt

python src/preprocessament.py
python src/eda.py
python src/preparacio_ml.py
python src/entrenament_models.py
python src/optimitzacio.py
python src/interpretabilitat.py
```
Requisit previ: cal col·locar el fitxer CSV original de l'EBR a data/raw/ (no s'inclou en aquest repositori per confidencialitat).

| Script | Genera | Carpeta de sortida |
|---|---|---|
| `preprocessament.py` | Dataset net (dates convertides a durades, sense outliers de format) | `data/processed/dataset_net.csv` |
| `eda.py` | Figures exploratòries (distribució, correlacions, dispersió) | `figures/01-03` |
| `preparacio_ml.py` | Split train/val/test, imputació, exploració de k | `data/processed/X_*_imp.csv`, `data/processed/y_*.csv` |
| `entrenament_models.py` | Comparació de 6 models baseline | `figures/04-05`, `models/best_model_baseline.pkl` |
| `optimitzacio.py` | Model final optimitzat (SVR) | `figures/06`, `models/model_final.pkl` |
| `interpretabilitat.py` | Permutation Importance, predicció vs real, residus | `figures/08-10` |

## Dades

El dataset conté 152 lots de producció amb 90 variables originals, incloent:

- Identificador de lot (OF)
- Dates d'inici i fi (Start / End) de cada operació (OP_1 a OP_20)
- Paràmetres de procés: pressió (Pa), agitació (rpm), temperatura de producte (ºC), càrregues d'ACN i altres materials (kg)
- Variable objectiu: Yield (%)

## Decisions metodològiques clau

- **Three-way split (80/10/10)** en lloc de l'habitual 80/20, per evitar contaminar el conjunt de test amb decisions de modelització (ajust d'hiperparàmetres, selecció d'algorisme).
- **Imputació de valors nuls (mediana)** ajustada únicament sobre el conjunt de train, per evitar data leakage.
- **Selecció de features (SelectKBest) i estandardització integrades dins d'un `Pipeline` de scikit-learn**, ajustades de manera independent a cada plec de la validació creuada. Això evita el data leakage que es produiria si la selecció de variables es fes una sola vegada abans de dividir en plecs (les dades de cada plec de "validació interna" no han d'influir mai en la tria de variables).
- **El nombre de features (k) es va determinar mitjançant una exploració amb validació creuada de 5 plecs** sobre el rang k=10 a k=30, seleccionant **k=19** com el valor amb millor R² mitjà (sense leakage).
- **5-fold cross-validation** sobre el conjunt de train per obtenir estimacions de rendiment més robustes.
- Les dates s'han transformat en durades (minuts) per a cada operació, ja que aporten més valor predictiu que les marques temporals absolutes.

## Model final

**SVR optimitzat** (RandomizedSearchCV, 19 features seleccionades), amb els resultats finals sobre el conjunt de test:

| Mètrica | Valor |
|---|---|
| R² | -0.0264 |
| MAE | 1.24% |
| RMSE | 1.49% |

Nota: el fitxer de dades original no s'inclou en aquest repositori per motius de confidencialitat de l'empresa.

**Autora:** Marta Guasch Segura
