# FallsInChianti
Official code repository for the manuscript: "Real-world mobility predicts falls in older adults." 
Holistic-fall-prediction/
│
├── README.md
├── requirements.txt
├── data/
│   └── sample_dummy_data.csv        <-- Data availability for the study
└── src/
    ├── utils.py                     <-- Custom transformers and CV splitters
    ├── 01_wfg_algorithm.py          <-- The WFG logic 
    ├── 02_univariate_analysis.py    <-- GEE models and Forest plots
    ├── 03_ml_pipeline.py            <-- Cleaned ML pipeline
    └── 04_explainability.py         <-- SHAP and UMAP visualizations
