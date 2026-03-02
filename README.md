# FallsInChianti
Official code repository for the manuscript: "Real-world mobility predicts falls in older adults." 

Holistic-fall-prediction/
│
├── README.md               # Descripción general e instrucciones
├── requirements.txt        # Dependencias necesarias para ejecutar el código
│
├── data/
│   └── sample_dummy_data.csv  # Datos sintéticos para demostración y pruebas
│
└── src/
    ├── utils.py            # Transformadores personalizados y divisores de validación cruzada (CV)
    ├── 01_wfg_algorithm.py  # Lógica del algoritmo Walking Feature Generation (WFG)
    ├── 02_univariate_analysis.py # Modelos GEE y generación de Forest plots
    ├── 03_ml_pipeline.py    # Pipeline completo de entrenamiento y evaluación de ML
    └── 04_explainability.py # Visualizaciones de SHAP y UMAP para interpretación

