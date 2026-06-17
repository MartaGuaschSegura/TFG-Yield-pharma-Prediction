TFG – Predicció del Yield en Producció Farmacèutica (Procés 5477-S03)
Treball de Fi de Grau – Enginyeria Biomèdica – Universitat Rovira i Virgili (URV)

Objectiu
Construir un model de Machine Learning capaç de predir el rendiment (Yield, %) d'un lot de producció farmacèutica a partir dels paràmetres de procés registrats (temperatures, pressions, agitació, càrregues d'ACN, etc.) al llarg de fins a 20 operacions seqüencials del procés 5477-S03.

Les dades provenen de l'Electronic Batch Record (EBR) implementat a l'empresa.

Estat actual del projecte
- Càrrega del dataset
- Neteja i preprocessament (conversió de dates a durades, separació de l'identificador de lot)
- Anàlisi exploratòria de dades (EDA)
- Preparació per a Machine Learning (split 80/10/10, imputació, estandardització)
- Entrenament i avaluació de models de regressió
- Optimització d'hiperparàmetres
- Interpretabilitat del model (Feature Importance / Permutation Importance)
- Visualitzacions finals
  
Estructura del repositori
- tfg-yield-prediction/
  - data/
    - raw/ ← CSV original de l'EBR (NO es puja, veure .gitignore)
    - processed/  ← Dades netes generades pel codi (NO es puja)
  - src/
    - preprocessament.py ← Càrrega + neteja + preprocessament
  - figures/ ← Figures generades automàticament pel codi
  - .gitignore
  - requirements.txt
  - README.md
    
Dades
El dataset conté 152 lots de producció amb 90 variables originals, incloent:
- Identificador de lot (OF)
- Dates d'inici i fi (Start / End) de cada operació (OP_1 a OP_20)
- Paràmetres de procés: pressió (Pa), agitació (rpm), temperatura de producte (ºC), càrregues d'ACN i altres materials (kg)
- Variable objectiu: Yield (%)

Decisions metodològiques clau
- Three-way split (80/10/10) en lloc de l'habitual 80/20, per evitar contaminar el conjunt de test amb decisions de modelització (ajust d'hiperparàmetres, selecció d'algorisme).
- 5-fold cross-validation sobre el conjunt de train per obtenir estimacions de rendiment més robustes.
- Imputació i estandardització ajustades (fit) únicament sobre el conjunt de train, per evitar data leakage.
- Les dates s'han transformat en durades (minuts) per a cada operació, ja que aporten més valor predictiu que les marques temporals absolutes.

Nota: el fitxer de dades original no s'inclou en aquest repositori per motius de confidencialitat de l'empresa.

Autora: Marta Guasch segura
