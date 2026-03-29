# Predicting PISA 2022 Math Scores with PySpark

A machine learning pipeline built in PySpark to predict student math proficiency scores using real data from the OECD's **Programme for International Student Assessment (PISA) 2022**, the world's largest international educational assessment (613,000+ students, 81 education systems).

## Results

| Metric | Value |
|--------|-------|
| R² | 0.40 |
| MAE | 63 pts |
| RMSE | 79.4 pts |

Predictions fall within one PISA proficiency level (≈ 75 pts) using only socioeconomic and demographic background variables.

**Top Predictors:**

| Feature | Importance |
|---------|-----------|
| Home Possessions (HOMEPOS) | 41.8% |
| ICT Resources | 18.4% |
| Socioeconomic Status (ESCS) | 17.9% |
| Country | 10.9% |
| Study Hours (Homework) | 5.9% |

Socioeconomic factors collectively account for 78% of predictive power.

## Project Structure
```
├── 01_PISA_Data_Acquisition.ipynb   # Downloads and cleans PISA 2022 data from OECD
├── 02_PISA_PySpark_Analysis.ipynb   # EDA, ML pipeline, model training, feature importance
├── .gitignore
└── README.md
```

## Data

The data is downloaded directly from the [OECD PISA 2022 Database](https://www.oecd.org/en/data/datasets/pisa-2022-database.html) via Notebook 1. No data files are included in this repo. Run pisa_data_downloader.ipynb to fetch and prepare the dataset.

**Variables used:**

| PISA Variable | Description | Type |
|---|---|---|
| `CNT` | Country / education system | Categorical |
| `ST004D01T` | Gender | Categorical |
| `MISCED` | Mother's education (ISCED) | Categorical |
| `FISCED` | Father's education (ISCED) | Categorical |
| `ESCS` | Economic, Social & Cultural Status index | Continuous |
| `HOMEPOS` | Home possessions index | Continuous |
| `ICTRES` | ICT resources index | Continuous |
| `STUDYHMW` | Weekly homework hours | Continuous |
| `PV1MATH` | Math plausible value (target) | Continuous |

## Pipeline

1. **Data Acquisition** — Download 2 GB SPSS file from OECD, extract 9 variables from 1,278, clean and export as CSV
2. **EDA** — Score distributions, feature relationships, correlation analysis across 81 education systems
3. **Feature Engineering** — PySpark ML Pipeline: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
4. **Modeling** — Linear Regression (baseline) and Random Forest (feature importance)
5. **Evaluation** — Test set metrics, residual diagnostics, country-level prediction accuracy

## Author

**Ethan Connelly**
- [LinkedIn](https://linkedin.com/in/ethan-connelly262)
- [GitHub](https://github.com/etconnelly)

B.S. Economics & B.S. Data Science, University of Wisconsin – Madison (2026)