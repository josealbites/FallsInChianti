# Real-world mobility predicts falls in older adults

Official code repository for the manuscript: **"An Explainable, Multi-Domain Digital Signature of Real-World Mobility Reveals Distinct Fall-Risk Phenotypes"**. This project provides a framework for predicting fall risk using inertial sensor data and machine learning.

## 📂 Project Structure

```text
Holistic-fall-prediction/
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
│
├── data/
│   └── sample_dummy_data.csv  # Data availability for the present study
│
└── src/
    ├── utils.py            # Custom transformers and CV splitters
    ├── 01_wfg_algorithm.py  # Walking Feature Generation (WFG) logic
    ├── 02_univariate_analysis.py # GEE models and Forest plots
    ├── 03_ml_pipeline.py    # End-to-end Machine Learning pipeline
    └── 04_explainability.py # SHAP and UMAP visualizations
