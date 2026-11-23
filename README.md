# PimaLens

**Leakage-free diabetes screening on the Pima Indians dataset.** The project audits where the data came from, then compares dimensionality reduction, feature selection and metaheuristic search under one nested cross-validation protocol.

The final model is logistic regression on **4 routine measurements** (pregnancies, 2-hour glucose, BMI, diabetes pedigree function):

| | Cross-validated (614 patients, 5 folds) | Locked test set (154 patients, opened once) |
|---|---|---|
| ROC-AUC | **0.846 ± 0.015** | **0.811** (95% CI 0.736–0.871) |
| Accuracy | **79.2% ± 1.9** | 69.5% at threshold 0.5 |
| Recall at screening threshold 0.235 | 85.0% | **88.9%** (48 of 54 diabetics) |

Built for **AID427 New Trends in Data Science (E-JUST), Project Phase 1**.

📄 **Full report:** [`docs/phase1_report.html`](docs/phase1_report.html) (download and open in a browser) · **Technical documentation:** [`Diabetes_Prediction_Pima_Indians_–_Phase_1_Technical_Documentation.md`](Diabetes_Prediction_Pima_Indians_–_Phase_1_Technical_Documentation.md)

---

## Why this project exists

An earlier version of this notebook reported **87.8% accuracy and 0.942 AUC**. The audit in this repository shows those numbers came from data leakage:

- The two input files were **the same 768 patients twice**, stacked into 1,144 rows. 56% of test rows had their own patient in the training set.
- The "clean" file had filled missing insulin with the **median of the patient's own diagnosis class** (102.5 if healthy, 169.5 if diabetic). For the 374 patients with missing insulin, that value alone predicts the outcome with **AUC 1.000**.
- The model and the decision threshold were chosen by looking at test-set results.

Re-estimated honestly with nested cross-validation, the old pipeline scores **75.9% accuracy and 0.829 AUC**. The corrected 79% cross-validated accuracy matches the 77–80% range of published single-classifier studies on this dataset.

## Key findings

| Question | Answer |
|---|---|
| Does nonlinear dimensionality reduction beat using all 8 features? | **No.** The best nonlinear method (Isomap, 4-D) loses 0.014 AUC. UMAP-2 loses 0.068. Across all 7 classifiers the best nonlinear change is −0.026 to +0.001 AUC. |
| Do GA and binary PSO find the exact best feature subset? | **Yes.** Exhaustive search over all 255 subsets gives the ground truth. Binary PSO reached it in **10/10 seeds** using ~25% of the evaluations. The GA reached it in 5/10, and random search in 4/10. |
| Which features matter? | Glucose, BMI and pregnancies appear in every good subset. **Insulin is not needed** once leaked values are removed, which reverses the old finding. |
| Is the final model overfitted? | No. Training AUC is 0.850 and out-of-fold AUC 0.847. The learning curve plateaus at ~0.845. Tree ensembles overfit (train AUC ~0.98) without improving validation AUC. |
| Why is test lower than CV? | Every pipeline drops 0.016–0.038 AUC on this particular 154-patient split. The drop comes from the split, not from overfitting. |

## What's inside

| Rubric area | Implementation |
|---|---|
| Data audit | OpenML checksum check, independent mirror comparison, reconstruction of the leaked file, missing-value profile |
| Preprocessing | Zeros → missing, standardisation, KNN imputation, all fitted inside each training fold |
| Linear reduction | PCA, LDA, truncated SVD |
| Nonlinear reduction | Kernel PCA, **from-scratch symmetric SNE (gradient-checked)**, t-SNE, UMAP, metric and non-metric MDS, Isomap, LLE, compared by trustworthiness at k = 5/10/30, runtime and class agreement |
| Clustering | K-means without labels, silhouette-selected k, compared with diagnosis by ARI/NMI |
| Feature selection | Mutual information (filter); forward, backward and RFECV (wrapper); L1, L2 and Elastic Net (embedded), each with an explicit selection rule |
| Metaheuristics | Genetic algorithm, **binary particle swarm optimisation** and random search, with equal budget, folds and objective; 10 seeds; exhaustive optimum as reference |
| Models | Logistic regression, KNN, SVM, naive Bayes, decision tree, random forest, gradient boosting; nested grid-search tuning |
| Evaluation | Locked 20% test, shared 5-fold outer CV, inner 5-fold CV for every choice, confusion matrices, all rubric metrics, bootstrap CIs, error analysis, learning curve |

## Repository layout

```
├── phase1.ipynb                     # executed notebook, the source of every number
├── docs/phase1_report.html          # designed technical report with embedded figures
├── Diabetes_Prediction_..._Documentation.md
├── data/
│   ├── pima_indians_diabetes.csv    # verified raw data (768 patients)
│   └── provenance/                  # OpenML ARFF + metadata, UCI mirror, pima.info
├── results/                         # CSV tables written by the notebook
├── archive/                         # the original notebook and documentation, kept as evidence
└── requirements.txt                 # exact package versions
```

## Reproduce

Requires Python 3.13. A full run takes about 10 minutes on a 20-thread CPU.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace phase1.ipynb --ExecutePreprocessor.timeout=-1
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m nbconvert --to notebook --execute --inplace phase1.ipynb --ExecutePreprocessor.timeout=-1
```

All randomness is seeded. Two consecutive runs produced identical tables; only runtimes differed.

## Data

The Pima Indians Diabetes Database comes from the National Institute of Diabetes and Digestive and Kidney Diseases and was donated to UCI by V. Sigillito in 1990: 768 women of Pima heritage aged 21 or older, 268 with diabetes. The copy used here is [OpenML dataset 37](https://www.openml.org/d/37). Its MD5 (`3cbaa3e54586aa88cf6aacb4033e4470`) is checked against the checksum OpenML publishes every time the notebook runs.

## Limitations

- The original leaked `diabetes.csv` was not available. Its construction is shown by reproducing the old notebook's statistics exactly, not by comparing bytes.
- The test set is small (154 patients), so test metrics carry wide intervals. The cross-validated estimates are more stable.
- One cohort of one population: no external validation.

## References

- Smith J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proc. SCAMC*, 261–265.
- Zou Q. et al. (2018). Predicting diabetes mellitus with machine learning techniques. *Front. Genet.* 9:515.
- Chang V. et al. (2022). Pima Indians diabetes mellitus classification based on machine learning algorithms. *Neural Comput. Appl.* doi:10.1007/s00521-022-07049-z.
- Hasan M.K. et al. (2020). Diabetes prediction using ensembling of different machine learning classifiers. *IEEE Access* 8, 76516–76531.
- Kennedy J., Eberhart R.C. (1997). A discrete binary version of the particle swarm algorithm. *IEEE SMC*.
- van der Maaten L., Hinton G. (2008). Visualizing data using t-SNE. *JMLR* 9.
