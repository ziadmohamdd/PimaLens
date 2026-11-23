## Diabetes Prediction (Pima Indians): Phase 1 Technical Documentation (revised)

**Team.** Faculty of Computer Science and Information Technology, Egypt-Japan University of Science and Technology (E-JUST), Alexandria, Egypt.

| Name | Student ID |
|---|---|
| Ziad Elsayed Fouad | 320230025 |
| Mohamed Adham | 320230001 |
| Mohamed Hussien | 320230008 |

Revised 2026-11-23 from a clean, top-to-bottom execution of `phase1.ipynb`. It replaces the version of 2026-11-21, which is kept unchanged in `archive/Technical_Documentation_original_2026-11-21.md`. The pre-audit notebook, with its original outputs, is in `archive/phase1_original_2026-11-23.ipynb`. Section numbers in square brackets, such as [§8], refer to notebook sections.

## 1. Executive summary

* **The historical headline numbers were not valid.** 87.8% accuracy, 0.942 AUC and the 0.35 threshold came from stacking two copies of the same 768 patients. One copy had its missing values filled with **class-conditional medians**, which encodes the outcome. 56% of the old test rows had their own patient in the training set. The model and the threshold were then chosen on the test set.
* **Rebuilt protocol.** One verified raw copy (768 patients). A stratified 20% test set (154) locked and opened once. One shared set of 5 stratified folds on the remaining 614 patients. All preprocessing, selection, reduction and tuning are fitted inside folds, with nested inner CV wherever a choice is made. The primary metric, ROC-AUC, was fixed in advance, and mean 5-fold accuracy is reported for the rubric.
* **Honest performance.** The locked finalist is logistic regression on 4 raw measurements (Pregnancies, Glucose, BMI, DiabetesPedigreeFunction) selected by binary PSO. Mean 5-fold CV: **AUC 0.846 ± 0.015, accuracy 0.792 ± 0.019**. Locked test: **AUC 0.811** (95% CI 0.736–0.871) and accuracy 0.695 at threshold 0.5. At the pre-specified screening threshold of 0.235: **recall 0.889**, specificity 0.630, accuracy 0.721.
* **Nonlinear reduction did not help.** For every one of the 7 classifiers, no nonlinear reduction with a valid out-of-sample transform beat using all 8 features (best change −0.026 to +0.001 AUC).
* **Metaheuristics work, but exhaustive search is the right tool for 8 features.** Under the same objective, budget and folds, binary PSO found the exact optimum in 10/10 seeds and the GA in 5/10, each using about 17–25% of the 255 evaluations that exhaustive search needs.

## 2. Problem and objectives

The task is to predict a positive diabetes test under WHO criteria from 8 routine measurements of Pima women aged 21 or older. This is a screening problem, where a missed diabetic costs more than a false alarm. The data is small (768), imbalanced (34.9% positive) and noisy: five measurements use `0` as a hidden missing-value code, and insulin is missing for 48.7% of patients [§0, §1.3].

## 3. Data provenance and version audit [§1]

### 3.1 Sources

Neither `diabetes.csv` nor `pima_indians_diabetes.csv` existed in the project folder at audit time. The raw data was re-obtained on 2026-11-23 and stored in `data/provenance/`:

| File | Source | Verification (all passed, asserted in the notebook) |
|---|---|---|
| `dataset_37_diabetes.arff` | OpenML data id 37 "diabetes" v1 (https://www.openml.org/d/37) | MD5 `3cbaa3e54586aa88cf6aacb4033e4470` equals the checksum OpenML publishes (`openml_37_meta.json`) |
| `jbrownlee_pima.csv` | Independent mirror of UCI `pima-indians-diabetes.data` | Identical to the ARFF row by row, including labels |
| `duke_pima.info` | UCI documentation `pima.info` | 768 instances, 500 / 268 classes, Smith et al. ADAP results |
| `data/pima_indians_diabetes.csv` | CSV export of the ARFF (columns `preg … class`) | Identical to the ARFF |

### 3.2 Were the two files different patients? No.

`diabetes.csv` could not be found, so the notebook rebuilds the hypothesised variant: each invalid zero is replaced by the median of the observed values in the patient's own Outcome class. The rebuild reproduces the old notebook's printed merged table exactly: 1,536 → **1,144** rows after de-duplication, and the count and mean of every column match to within 5 × 10⁻⁷ [§1.2]. This strongly supports the hypothesis, but byte-level identity with the original file **could not be verified**.

* The never-imputed columns (Pregnancies, DiabetesPedigreeFunction, Age, Outcome) agree for all 768 rows, and every observed value agrees. The 392 complete-case patients are exact copies, which is why exactly 392 rows were removed as duplicates.
* Each class gets exactly one fill value (Insulin: 102.5 for Outcome 0, 169.5 for Outcome 1; similarly for the other four columns). Among the 374 patients with missing insulin, **the filled insulin alone predicts the outcome with AUC 1.000**. Genuine insulin on the 394 observed patients gives AUC 0.732.
* In the historical split, 376 patients appeared twice, and **128 of the 229 test rows (55.9%) had their own patient in training**.

**Decision.** Use the raw file only, one row per patient. The clean variant is never used for modelling.

### 3.3 Raw data profile (768 patients)

| | Count | Share |
|---|---|---|
| Outcome 0 / 1 | 500 / 268 | 65.1% / 34.9% |
| Insulin = 0 (missing) | 374 | 48.7% |
| SkinThickness = 0 | 227 | 29.6% |
| BloodPressure = 0 | 35 | 4.6% |
| BMI = 0 | 11 | 1.4% |
| Glucose = 0 | 5 | 0.7% |
| Complete cases | 392 | 51.0% |

Pregnancies = 0 (111 patients) is a legitimate value and is kept.

## 4. Evaluation protocol [§0, §2]

| Element | Choice |
|---|---|
| Test set | Stratified 20% (154 patients, 54 positive), `random_state=42`, held in a `TestVault` object that can be opened only once [§2.1] |
| Development set | 614 patients (214 positive) |
| Outer folds | One `StratifiedKFold(5, shuffle, 42)` split reused by every comparable experiment |
| Inner CV | `StratifiedKFold(5, shuffle, 7)` inside each outer training fold, for tuning, wrappers, embedded C selection, MI-k selection and metaheuristic searches |
| Preprocessing | Keep requested raw columns → zeros to NaN → `StandardScaler` → `KNNImputer(5)`, fitted per fold [§2.2] |
| Unsupervised reducers | Wrapped so the outcome is never passed to `fit`. Without the wrapper, scikit-learn would pass `y` to UMAP and silently run supervised UMAP |
| Primary metric | Mean outer 5-fold ROC-AUC; mean 5-fold accuracy is always reported alongside |
| Finalist rule | Highest mean CV AUC; ties within 0.001 go to fewer raw measurements, then the simpler model |
| Threshold rule | Highest threshold with out-of-fold recall ≥ 0.85 on the development set |
| Parsimony unit | A feature must earn ≥ 0.002 AUC (SFS tolerance, MI-k rule, metaheuristic penalty) |

## 5. Baseline: 7 classifiers on all 8 features [§3]

Mean 5-fold CV on the development set:

| Classifier | Accuracy (mean ± SD) | ROC-AUC | Train AUC | Train − CV AUC |
|---|---|---|---|---|
| Logistic regression | 0.788 ± 0.020 | 0.844 | 0.852 | 0.008 |
| Random forest (200, depth 6) | 0.767 ± 0.017 | 0.836 | 0.977 | 0.141 |
| Gaussian naive Bayes | 0.757 ± 0.026 | 0.831 | 0.838 | 0.007 |
| SVM (RBF) | 0.764 ± 0.030 | 0.831 | 0.905 | 0.074 |
| Gradient boosting (150, depth 3) | 0.752 ± 0.008 | 0.820 | 0.982 | 0.162 |
| KNN (k = 7) | 0.754 ± 0.043 | 0.810 | 0.893 | 0.083 |
| Decision tree (depth 5) | 0.753 ± 0.034 | 0.787 | 0.921 | 0.133 |

The ranking reverses the historical one. Without leaked insulin values and duplicated patients, the tree ensembles no longer lead. They overfit (train AUC around 0.98), while logistic regression generalises best.

## 6. Dimensionality reduction

### 6.1 Linear [§4]

PCA on the standardised development data: PC1 32.2% and PC2 19.7% of variance; 5 components reach 80% and 6 reach 90%. Truncated SVD gives the same ratios because the features are already almost centred (max |column mean| 0.011). Binary LDA has exactly one axis; its largest weights are Glucose 0.91, BMI 0.49 and Pregnancies 0.28.

### 6.2 Custom SNE verified [§5.1]

The from-scratch SNE is **symmetric SNE**: joint $p_{ij} = (p_{j|i}+p_{i|j})/2n$ and a joint Gaussian $q_{ij} \propto \exp(-\lVert y_i-y_j\rVert^2)$. It is not the original asymmetric SNE. Three checks pass:

* achieved perplexities equal the target (30.000);
* the analytic gradient $4\sum_j (p_{ij}-q_{ij})(y_i-y_j)$ matches a finite-difference gradient of the KL objective (relative error 2 × 10⁻¹⁰);
* KL falls from 2.86 to 1.33 over 1,000 iterations.

### 6.3 Embedding quality: 11 methods, same 614 patients, 2-D [§5.2]

| Method | Trustworthiness k=5 / 10 / 30 | 10-NN label agreement | Seconds | Out-of-sample transform |
|---|---|---|---|---|
| t-SNE | 0.977 / **0.958** / 0.912 | 0.694 | 0.7 | no |
| UMAP | 0.947 / 0.935 / 0.900 | 0.693 | 8 | yes |
| SNE (custom symmetric) | 0.856 / 0.858 / 0.861 | 0.674 | 13 | no |
| Metric MDS | 0.848 / 0.855 / 0.870 | 0.677 | 0.6 | no |
| Non-metric MDS | 0.851 / 0.855 / 0.870 | 0.672 | 3 | no |
| Isomap (k=15) | 0.827 / 0.833 / 0.842 | 0.668 | 0.05 | yes |
| PCA | 0.831 / 0.831 / 0.844 | 0.675 | <0.01 | yes |
| Truncated SVD | 0.831 / 0.831 / 0.844 | 0.675 | <0.01 | yes |
| Kernel PCA (RBF, γ=0.1) | 0.793 / 0.798 / 0.804 | 0.665 | 0.01 | yes |
| LLE (k=15) | 0.773 / 0.774 / 0.787 | 0.668 | 0.07 | yes |
| LDA (supervised, 1-D) | 0.657 / 0.661 / 0.675 | 0.710 | <0.01 | yes |

The 8-D reference label agreement is 0.696; the majority-class rate is 0.652.

**Visual separation vs reality.** t-SNE and UMAP preserve neighbourhoods best. No unsupervised embedding increases class agreement above the original 8-D space, so the class overlap visible in the plots is genuine. Only LDA, which was fitted with the labels, shows more separation. SNE trails t-SNE by about 0.10 in trustworthiness at k = 10, which is the crowding problem.

### 6.4 Clustering without labels [§6]

K-means (silhouette-chosen k = 2, silhouette 0.21) splits the patients into an older, heavier, higher-glucose cluster (diabetes rate 58%, computed post hoc) and a younger, leaner one (16%). Agreement with diagnosis is weak: **ARI 0.19, NMI 0.15**. 118 of the 279 patients in the "high-risk" cluster are not diabetic. The clusters are risk profiles, not the diagnosis. Before/after plots: raw Glucose–BMI, PCA coloured by the known outcome, PCA and UMAP coloured by cluster.

## 7. Feature selection

### 7.1 Filter, wrapper and embedded rules and their dev-set selections [§7]

| Method | Rule | Selected on full development set |
|---|---|---|
| Mutual information | Smallest top-k within 0.002 of the best inner-CV AUC | all 8 (k = 8) |
| Forward SFS | Add features while the gain is ≥ 0.002 | Pregnancies, Glucose, BMI, DPF |
| Backward SFS | Remove features while the loss is ≤ 0.002 | Pregnancies, Glucose, BMI, DPF |
| RFECV | Size with the best inner-CV AUC | 6 (drops SkinThickness, Insulin) |
| L1 | C by inner-CV AUC (C = 1); non-zero weights | 6 (drops SkinThickness, Insulin) |
| L2 | C by inner-CV AUC (C = 0.1); \|coef\| ≥ mean \|coef\| | Pregnancies, Glucose, BMI |
| Elastic Net (l1_ratio 0.5) | C by inner-CV AUC (C = 0.46); non-zero weights | 6 (drops SkinThickness, Insulin) |

These methods select **after** imputation, so a dropped column still contributes through the imputer. They do not reduce the number of measurements a clinic must take.

### 7.2 Metaheuristics vs exact optimum [§8]

Objective: mean 5-fold CV AUC of (raw-column subset → zero recoding → scale → KNN impute → estimator) − 0.002 × |S|. Columns are dropped before imputation. The GA keeps the original operators (2-way tournament, single-point crossover 0.9, bit-flip 1/8, elitism). Binary PSO uses the Kennedy–Eberhart sigmoid transfer with constriction coefficients (w = 0.729, c₁ = c₂ = 1.494, |v| ≤ 4). Every algorithm gets the same budget of 80 calls, and repeated subsets are cached. Ten seeds per algorithm.

| Objective | Algorithm | Hit exact optimum | Mean gap | Unique subsets (cost) | Unique evals to final best | Distinct final subsets |
|---|---|---|---|---|---|---|
| LR | **Binary PSO** | **10/10** | 0.0000 | 63.0 | 48.3 | 1 |
| LR | GA | 5/10 | 0.0009 | 44.3 | 21.6 | 5 |
| LR | Random search | 4/10 | 0.0007 | 69.8 | 34.7 | 5 |
| LR | Exhaustive | exact | 0 | 255 (13.5 s) | n/a | 1 |
| GB | Binary PSO | 6/10 | 0.0010 | 62.7 | 43.4 | 4 |
| GB | GA (original tree-GA) | 4/10 | 0.0022 | 45.4 | 29.7 | 5 |
| GB | Random search | 2/10 | 0.0024 | 69.8 | 32.0 | 4 |
| GB | Exhaustive | exact | 0 | 255 (~200 s on one core) | n/a | 1 |

Exact optima:

* **LR:** {Pregnancies, Glucose, BMI, DiabetesPedigreeFunction}, AUC 0.8456.
* **GB:** {Pregnancies, Glucose, Insulin, BMI, Age}, AUC 0.833.

The top of the LR landscape is flat (10 subsets within 0.004). **Insulin and SkinThickness are not needed by the LR objective.** This reverses the historical finding that Insulin was "selected by every method", which came from the leaked insulin values.

**Nested.** Inside each outer fold (one seed, 80 calls), the GA reached the inner exhaustive optimum in 3/5 folds and PSO in 2/5. Every miss was within 0.003, and outer AUC was unchanged within noise [§9 d].

## 8. Prediction comparison on identical folds [§9]

Reference classifier is logistic regression. Δ is relative to all 8 features; "folds better" counts the 5 outer folds.

| Family (best member) | Dims | CV AUC | Δ AUC | Folds better | CV accuracy |
|---|---|---|---|---|---|
| All 8 original features | 8 | 0.844 | 0 | n/a | 0.788 |
| Metaheuristic raw selection (binary PSO) | 4.2 | 0.846 | +0.002 | 4 | 0.792 |
| Embedded (L1) | 7.0 | 0.844 | +0.000 | 1 | 0.787 |
| Wrapper (forward SFS) | 4.0 | 0.844 | +0.000 | 3 | 0.793 |
| Linear (LDA, 1-D supervised) | 1 | 0.842 | −0.001 | 1 | 0.787 |
| Filter (MI) | 6.8 | 0.842 | −0.002 | 1 | 0.788 |
| **Nonlinear (Isomap, 4)** | 4 | **0.830** | **−0.014** | 1 | 0.766 |

Other nonlinear representations: Kernel PCA-4 −0.018, LLE-4 −0.025, UMAP-4 −0.036, UMAP-2 −0.068, LLE-2 −0.077 (LLE is unstable across folds).

Across all 7 classifiers, the best nonlinear representation changes AUC relative to all 8 features by −0.026 to +0.001, so the conclusion does not depend on the model. SNE, t-SNE and both MDS variants were left out of prediction because they have no out-of-sample mapping.

## 9. Nested tuning and the historical pipeline [§10]

| Pipeline (whole pipeline inside `GridSearchCV`, inner 5-fold) | CV AUC | CV accuracy | Train − CV AUC |
|---|---|---|---|
| Logistic regression (C tuned; chose 0.1–0.32) | 0.844 | 0.787 | 0.007 |
| SVM RBF (C, γ tuned) | 0.844 | 0.787 | 0.024 |
| Random forest (tuned) | 0.837 | 0.775 | 0.081 |
| Gradient boosting (historical 108-point grid) | 0.834 | 0.770 | 0.088 |
| KNN (tuned) | 0.830 | 0.757 | 0.170 |
| **Historical replica**: tree-GA subset re-searched per fold + tuned GB | **0.829** | **0.759** | 0.115 |

Re-estimated honestly, the historical "87.8% / 0.942" pipeline scores **0.759 accuracy and 0.829 AUC**.

## 10. Finalist and test results [§11, §12]

181 pipelines were compared on identical folds. The top is **binary-PSO-selected raw features + logistic regression** (CV AUC 0.8458 ± 0.0153, accuracy 0.7915 ± 0.0186). The next best are tuned LR (0.8444), LR on the tree-GA subset (0.8444) and tuned SVM (0.8440). 18 pipelines fall inside the one-SE band (SE 0.0068). Refitted on the full development set, PSO chose Pregnancies, Glucose, BMI and DiabetesPedigreeFunction.

The screening threshold, set from out-of-fold development predictions, is **0.235**: OOF recall 0.851, specificity 0.620.

| Locked test (154: 100 negative, 54 positive) | Threshold 0.5 | Threshold 0.235 |
|---|---|---|
| Confusion matrix (TN, FP, FN, TP) | 80, 20, 27, 27 | 63, 37, 6, 48 |
| Accuracy / error rate | 0.695 / 0.305 | 0.721 / 0.279 |
| Precision | 0.574 | 0.565 |
| Recall | 0.500 | 0.889 |
| Specificity | 0.800 | 0.630 |
| F1 | 0.535 | 0.691 |
| ROC-AUC (95% bootstrap CI) | 0.811 (0.736–0.871) | 0.811 |

**Errors.** False negatives at 0.5 have median glucose 123, versus 164 for true positives; they are younger and leaner. The screening threshold rescues 21 of the 27 at the cost of 17 more false alarms. Of the 6 diabetics still missed, 4 lack insulin and/or skin-fold values, and their glucose is 78–129. False positives look diabetic (median glucose 153.5, BMI 37.5).

**Over/underfitting.** Train AUC 0.850, out-of-fold 0.847, test 0.811. The learning curve plateaus at validation AUC 0.845, a high-bias regime limited by the information in the features. The flexible models overfit without gaining validation AUC.

**Why test is below CV.** A post-hoc check, run after locking and unable to change the choice, refitted four other pipelines. All lose 0.016–0.038 AUC on this split; the finalist loses 0.035. The test split is harder than average, and the result lies within the bootstrap interval.

## 11. Related work [§13]

| Study | Data | Protocol | Reported | Comparability limits |
|---|---|---|---|---|
| Smith et al., 1988 (ADAP). Verified via UCI `pima.info` and the PMC2245318 abstract | 768 | 576 train / 192 test | Sensitivity = specificity = 0.76 at cutoff 0.448 | One split, no AUC; full text is a scanned PDF |
| Zou et al., 2018, *Front. Genet.* 9:515 (PMC6232260) | 392 complete cases | 5-fold CV (10-fold in their Table 8) | RF + mRMR (glucose, insulin, age): accuracy 0.7721 (0.7852 in Table 8) | Drops incomplete patients; accuracy only |
| Chang et al., *Neural Comput. Appl.*, online 2022 (doi 10.1007/s00521-022-07049-z, PMC8943493) | 768, median-imputed | One 70/30 split (538/230) | RF accuracy 79.57%, AUC 86.24%; NB on 3 features 79.13% | Single split; some RF metrics do not follow from the paper's own confusion matrices |
| Hasan et al., 2020, *IEEE Access* 8:76516 (abstract only) | Pima | K-fold CV + grid search | Weighted ensemble AUC 0.950, sensitivity 0.789, specificity 0.934 | Full text not retrievable: fold count and nesting unverified; outlier rejection changes the patient set |
| This work | 768 verified | Locked test + nested 5-fold | CV AUC 0.846 / acc 0.792; test AUC 0.811 / acc 0.695 | n/a |

The corrected CV accuracy (0.79) is in line with the 0.77–0.80 range of the verified single-classifier studies. The historical 0.878 was 8–10 points above that range, which is consistent with the leakage found in §3.2.

## 12. Reconciliation with the 2026-11-21 version

| Historical claim | Status after this run |
|---|---|
| "Two public versions … combined into one reliable table" (1,144 rows) | **Withdrawn.** They are the same 768 patients, and one version leaks the label |
| Best model: GB on {Glucose, BloodPressure, Insulin, BMI, Age}, accuracy 0.878, AUC 0.942 | **Withdrawn.** The nested re-estimate of that pipeline is accuracy 0.759, AUC 0.829 |
| Threshold 0.35, recall 0.938 | **Replaced.** It was chosen on the test set. The new threshold is 0.235, chosen out-of-fold; test recall 0.889 |
| "Gradient boosting and random forest lead every ranking" | **Reversed.** Logistic regression leads; tree ensembles overfit |
| "Insulin and BMI are selected by every method" | **Reversed for Insulin**, which no LR-based method needs. BMI and Glucose remain universal |
| Exhaustive best subset: all except BloodPressure and Age (AUC 0.8577) | **Replaced** by the raw-column penalised optimum {Pregnancies, Glucose, BMI, DPF}, AUC 0.8456 |
| Isomap and LLE with 10 neighbours | The code used 15; the documentation now matches the code |
| "No quantitative embedding-quality score" | **Added.** Trustworthiness at k = 5/10/30, label agreement, runtime |
| Imputation outside CV; single split; no intervals | **Fixed.** In-fold pipelines, shared folds, nested CV, bootstrap CIs |
| Kapoor & Narayanan 2023 (leakage survey) | Not re-verified in this audit; kept out of the related-work table |

## 13. Reproducibility

* **Rerun** from the project folder in PowerShell. It takes about 10 minutes on this machine (20 logical cores):
  `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace phase1.ipynb --ExecutePreprocessor.timeout=-1`
  Use `python -m nbconvert` from the venv rather than `python -m jupyter nbconvert`: on this machine the latter dispatches to the system Python 3.14 install.
* **Environment.** `.venv` (Python 3.13.9, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.3.5, umap-learn 0.6.0). Kernel `aid427-phase1` is registered inside the venv. Exact versions are in `requirements.txt`.
* **Determinism.** Two consecutive full runs gave identical tables; only wall-clock times differed.
* **Outputs.** Key tables are written to `results/`: leaderboard, representation × classifier, metaheuristic seeds, embedding quality, final test metrics, post-hoc diagnostic, reconciliation.

## 14. What could not be verified

* The original `diabetes.csv` is not available, so its class-median construction is established by exact reproduction of the old outputs, not by byte comparison.
* Hasan et al. 2020 was checked from the abstract only; the publisher blocked the full text.
* The body of Smith et al. 1988 is a scanned PDF. Its numbers come from the UCI documentation and the abstract.
* Chang et al.'s volume and page numbers were not confirmed; the PMC copy is the online-ahead-of-print version.
* With 154 test patients, test metrics have wide intervals (accuracy ± about 7 points), and the cross-validated estimates are the more stable numbers.

## 15. References

1. Smith J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proc. SCAMC*, 261–265. PMC2245318.
2. UCI Pima Indians Diabetes Database documentation (`pima.info`); OpenML data id 37.
3. Zou Q. et al. (2018). Predicting diabetes mellitus with machine learning techniques. *Front. Genet.* 9:515. PMC6232260.
4. Chang V., Bailey J., Xu Q.A., Sun Z. Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms. *Neural Comput. Appl.* (online 2022). doi:10.1007/s00521-022-07049-z.
5. Hasan M.K. et al. (2020). Diabetes prediction using ensembling of different machine learning classifiers. *IEEE Access* 8, 76516–76531. doi:10.1109/ACCESS.2020.2989857.
6. van der Maaten L., Hinton G. (2008). Visualizing data using t-SNE. *JMLR* 9, 2579–2605.
7. Hinton G., Roweis S. (2002). Stochastic neighbor embedding. *NIPS* 15.
8. McInnes L., Healy J., Melville J. (2018). UMAP. arXiv:1802.03426.
9. Kennedy J., Eberhart R.C. (1997). A discrete binary version of the particle swarm algorithm. *IEEE SMC*.
10. Venna J., Kaski S. (2001). Neighborhood preservation in nonlinear projection methods. *ICANN*.
11. Holland J.H. (1975). *Adaptation in Natural and Artificial Systems*.
12. Pedregosa F. et al. (2011). Scikit-learn. *JMLR* 12, 2825–2830.

References 6–12 are standard method citations given from bibliographic knowledge; they were not re-opened in this audit.
