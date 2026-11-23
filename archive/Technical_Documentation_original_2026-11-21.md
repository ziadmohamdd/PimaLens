## Diabetes Prediction (Pima Indians) – Phase 1 Technical Documentation

Nov 21, 2026 · @Mihamed

## 1. Executive summary

Phase 1 delivers a complete preprocessing, feature-engineering and benchmarking pipeline for diabetes prediction on two merged copies of the Pima Indians dataset, and its best model reaches 87.8% accuracy and 0.942 ROC-AUC on a 229-patient hold-out set. Those numbers should be read as upper bounds, not as expected real-world performance

(see 1.2).

## 1.1 What was built

- Data: two 768-row Pima files merged to 1,536 rows, de-duplicated to 1,144, with invalid zeros recoded as missing and filled by KNN imputation fitted on the training split only.

- Representation: 3 linear and 8 non-linear dimensionality-reduction methods, used for visual inspection of class separability.

- Feature selection: forward, backward, RFECV, exhaustive search, L1/L2/Elastic-Net and two genetic algorithms (one logistic-regression-driven, one gradient-boosting- driven).

- Models: 7 classifiers benchmarked on 3 feature sets; the winner (Gradient Boosting on 5 GA-selected features) was grid-search tuned and given a screening threshold.

## 1.2 Headline results and the key caveat

- Best balanced model: Gradient Boosting on {Glucose, BloodPressure, Insulin, BMI, Age}: accuracy 0.878, F1 0.825, AUC 0.942.

- Screening configuration (threshold 0.35): recall 0.938 with 5 missed cases out of 81 positives, at the cost of 28 false alarms among 148 negatives.

- Caveat: the merge most likely places the same patient in both the training and test sets, and the 'clean' file appears to carry label-derived Insulin values. Both effects inflate the scores above and must be fixed before any claim is made (Section 9).


## 2. Introduction and objectives

Diabetes mellitus is a chronic metabolic disease, and early screening from routine clinical measurements is a long-standing target for machine learning [6]. The Pima Indians Diabetes dataset, first used to evaluate the ADAP neural-network algorithm in 1988 [1], remains the most common public benchmark for this task.

Phase 1 answers four questions, in order:

- 1. Can the two public versions of the dataset be combined and cleaned into one reliable table?

- 2. Does the data show class structure under linear or non-linear projection?

- 3. Which of the 8 clinical features carry the predictive signal, according to several independent selection strategies?

- 4. Which classifier and decision threshold give the best trade-off for a screening use case, where a missed diabetic is costlier than a false alarm?

The work is implemented in a single Python notebook (pandas, scikit-learn, UMAP, seaborn) with a fixed random seed of 42 throughout.

## 3. Dataset and integration

The source is the NIDDK Pima Indians Diabetes database: 768 women of Pima heritage, aged 21 or older, 500 negative and 268 positive for diabetes under WHO criteria [2]. Phase 1 uses two files of this data, diabetes.csv (a pre-cleaned variant) and pima_indians_diabetes.csv (the raw variant with text labels).

| Feature | Meaning | Unit |
| --- | --- | --- |
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose, 2 h in an oral glucose tolerance | mg/dL |
|   | test |   |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin-fold thickness | mm |
| Insulin | 2-hour serum insulin | mu U/ml |
| BMI | Body-mass index | kg/m² |


| Feature | Meaning | Unit |
| --- | --- | --- |
| DiabetesPedigreeFunction Family-history risk score |   | index |
| Age | Age | years |
| Outcome | Diabetes diagnosis (target) | 0 / 1 |

## Integration steps

- 1. Rename the raw file's columns ( preg , plas , pres , skin , insu , mass , pedi , age , class ) to the clean file's names.

- 2. Map tested_positive / tested_negative to 1 / 0.

- 3. Concatenate vertically: 1,536 rows × 9 columns, saved as combined_diabetes.csv .

Because 1,536 = 2 × 768, the two files describe the same underlying patients rather than two cohorts. This drives the main validity concern in Section 9.

## 4. Preprocessing and exploratory analysis

The pipeline fits every learned transform (scaler, imputer) on the training split only, which is the correct order for avoiding preprocessing leakage.

- 1. De-duplicate: exact duplicate rows removed, 1,536 → 1,144 rows.

- 2. Recode invalid zeros: zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible and become missing values.

- 3. Stratified split: 80/20 with random_state=42 , giving 915 training and 229 test patients.

- 4. Scale: StandardScaler fitted on the training set, so that distance-based imputation weighs all features equally.

- 5. Impute: KNNImputer with 5 neighbours [8], fitted on the training set and applied to the test set. No missing values remain in either split.

- 6. Inverse-scale the cleaned training data to original units for plots.

## Missingness after de-duplication (1,144 rows)

| Feature | Missing Share |   |
| --- | --- | --- |
| Insulin | 374 | 32.7% |


| Feature | Missing Share |
| --- | --- |
| SkinThickness 227 | 19.8% |
| BloodPressure 35 | 3.1% |
| BMI | 11 1.0% |
| Glucose | 5 0.4% |

The positive class is 35.5% of the data (mean Outcome 0.355), a moderate imbalance handled by stratification rather than resampling. The exploratory plots are a class-balance bar chart, per-feature histograms, a correlation heatmap and Glucose / BMI / Age / Insulin boxplots split by outcome.

## 5. Dimensionality reduction

Two linear projections capture only about half of the variance in two dimensions (49.3%), and the notebook uses all projections for visual inspection of class separability, not as classifier inputs.

Linear methods (fitted on the cleaned, scaled training set)

| Method | Setting | Result |
| --- | --- | --- |
| PCA [9] | 2 components | PC1 30.07% + PC2 19.19% = 49.26% variance |
| LDA [9] | 1 component (classes − | Supervised 1-D axis; class densities plotted as |
|   | 1) | KDE |
| Truncated | 2 components | Identical to PCA (49.26%) |
| SVD |   |   |

SVD matches PCA exactly because the inputs are already standardised to zero mean; the two methods differ only on uncentred data, so SVD adds no information here.

Non-linear methods (all 2-D, plotted side by side)

- Kernel PCA with an RBF kernel (gamma 0.1) [10]

- Symmetric SNE, implemented from scratch in NumPy with a Gaussian low- dimensional kernel [11]

- t-SNE, perplexity 30 [12]


- UMAP, 15 neighbours, min_dist 0.1 [13]

- Metric and non-metric MDS

- Isomap, 10 neighbours [14]

- Locally Linear Embedding, 10 neighbours [15]

The custom SNE function exists to contrast a Gaussian with a Student-t low-dimensional kernel, which is the exact difference between SNE and t-SNE. No quantitative embedding-quality score (trustworthiness, stress) is recorded, so the comparison is qualitative.

## 6. Feature selection

Insulin and BMI are selected by every method tested, Glucose by all but forward selection, and BloodPressure by almost none of the linear methods, yet the tree-based search keeps it.

| Strategy | Type | Selected features |
| --- | --- | --- |
| Forward selection (k = 4, ROC- | Wrapper | SkinThickness, Insulin, BMI, |
| AUC, 5-fold) |   | DiabetesPedigreeFunction |
| Backward elimination (k = 4) Wrapper |   | Pregnancies, Glucose, Insulin, BMI |
| RFECV [16] | Wrapper | 7 features: all except BloodPressure |
| Exhaustive search (255 | Wrapper | 6 features, CV ROC-AUC 0.8577: all |
| subsets) |   | except BloodPressure and Age |
| L1 / Lasso, C = 0.2 [17] | Embedded | 7 features: drops BloodPressure |
| L2 / Ridge and Elastic Net 50/50 | Embedded All 8 (BloodPressure shrunk to −0.058 |   |
| [18] |   | and −0.027) |
| Genetic algorithm, logistic | Evolutionary Pregnancies, Glucose, SkinThickness, |   |
| regression fitness |   | Insulin, BMI |
| Genetic algorithm, gradient | Evolutionary Glucose, BloodPressure, Insulin, BMI, Age |   |
| boosting fitness |   |   |

Genetic-algorithm design. Binary chromosome of length 8 [21]; fitness is 5-fold ROC- AUC minus a per-feature penalty (0.002 for the logistic version, 0.001 for the boosting version); tournament selection, single-point crossover, bit-flip mutation (0.10 and 0.12),


and elitism. Population 20 over 15 generations for the logistic version and 16 over 12 for the boosting version.

Coefficient view. Under L1 regularisation Glucose carries by far the largest standardised weight (0.90), followed by Insulin (0.50), Pregnancies (0.39), BMI (0.38) and SkinThickness (0.32).

The disagreement on BloodPressure suggests it contributes through interactions that a linear model cannot use. The strong, universal weight on Insulin needs scrutiny, because a third of its values were imputed (Section 9).

## 7. Modelling, evaluation and optimisation

Gradient Boosting and Random Forest lead every ranking, with ROC-AUC between 0.926 and 0.943, while Logistic Regression and Naive Bayes trail at 0.84 to 0.85.

## 7.1 Benchmark design

Seven classifiers were trained on three feature sets (all 8 features, the 5 logistic-GA features, the 5 boosting-GA features) and scored on the 229-patient hold-out set: Logistic Regression, KNN (k = 7), RBF-SVM, Gaussian Naive Bayes, Decision Tree (depth 5), Random Forest (200 trees, depth 6) and Gradient Boosting (150 trees, learning rate 0.05, depth 3) [19, 20]. Metrics are accuracy, error rate, precision, recall, F1, ROC-AUC and the confusion matrix. The hold-out set contains 148 negatives and 81 positives.

## 7.2 Results

Top 8 of 23 configurations by ROC-AUC (confusion matrix as [[TN, FP], [FN, TP]]):

| Feature set | Model | Confusion | Accuracy Precision Recall F1 |   | ROC- |
| --- | --- | --- | --- | --- | --- |
|   |   | matrix |   |   | AUC |
| All 8 | Gradient | [[131, 17], [16, | 0.856 | 0.793 | 0.803 0.798 0.943 |
|   | Boosting | 65]] |   |   |   |
| Tree-GA, tuned, | Gradient | [[120, 28], [5, | 0.856 | 0.731 | 0.938 0.822 0.942 |
| threshold 0.35 | Boosting | 76]] |   |   |   |
| Tree-GA, tuned, | Gradient | [[130, 18], [13, | 0.865 | 0.791 | 0.840 0.814 0.942 |
| threshold 0.50 | Boosting | 68]] |   |   |   |
| Tree-GA (5) | Gradient | [[135, 13], [15, | 0.878 | 0.835 | 0.815 0.825 0.942 |
|   | Boosting | 66]] |   |   |   |


| Feature set | Model | Confusion | Accuracy Precision Recall F1 |   | ROC- |
| --- | --- | --- | --- | --- | --- |
|   |   | matrix |   |   | AUC |
| Tree-GA (5) | Random | [[128, 20], [11, | 0.865 | 0.778 | 0.864 0.819 0.941 |
|   | Forest | 70]] |   |   |   |
| Logistic-GA (5) | Gradient | [[130, 18], [21, | 0.830 | 0.769 | 0.741 0.755 0.932 |
|   | Boosting | 60]] |   |   |   |
| All 8 | Random | [[127, 21], [16, | 0.838 | 0.756 | 0.803 0.778 0.930 |
|   | Forest | 65]] |   |   |   |
| Logistic-GA (5) | Random | [[129, 19], [15, | 0.852 | 0.777 | 0.815 0.795 0.926 |
|   | Forest | 66]] |   |   |   |

## Three patterns stand out:

- Model-matched selection works. The boosting-driven GA kept Gradient Boosting at 0.942 AUC with 3 fewer features, while the logistic-driven GA lost about 0.011 AUC for the same model.

- Distance and kernel models gain from fewer features. With the boosting-GA subset KNN rises from 0.870 to 0.908 AUC, SVM from 0.896 to 0.908 and Decision Tree from 0.885 to 0.902.

- Linear and probabilistic models do not. Logistic Regression stays at 0.852 or below and Naive Bayes at 0.845 or below on every subset.

## 7.3 Hyperparameter tuning and threshold selection

Grid search (5-fold, ROC-AUC) over 108 combinations of trees, learning rate, depth, subsample and leaf size selected n_estimators=150 , learning_rate=0.05 ,

max_depth=2 , min_samples_leaf=3 , subsample=1.0 , with a cross-validated AUC of

0.920.

Lowering the decision threshold trades false alarms for missed cases:

| Threshold Accuracy Precision Recall F1 Missed (FN) False alarms (FP) |   |   |   |   |
| --- | --- | --- | --- | --- |
| 0.50 | 0.865 | 0.791 | 0.840 0.814 13 | 18 |
| 0.40 | 0.852 | 0.733 | 0.914 0.813 7 | 27 |
| 0.35 | 0.856 | 0.731 | 0.938 0.822 5 | 28 |
| 0.30 | 0.856 | 0.726 | 0.951 0.824 4 | 29 |


The notebook chooses 0.35: it removes 8 of the 13 missed cases at the price of 10 extra false alarms, with accuracy essentially unchanged.

## 7.4 How much weight the differences carry

With 229 test patients the standard error of an accuracy near 0.86 is about 2.3 percentage points, so gaps of 1 to 2 points, and AUC gaps below about 0.01 among the top five rows, are within noise. Only the separation between the tree ensembles (AUC above 0.92) and the linear and Naive Bayes models (below 0.86) is large enough to be meaningful, and it too is affected by the issues in Section 9.

## 8. Related work and comparison

Published work on the Pima data reports roughly 76% to 80% accuracy for single classifiers and up to 0.95 AUC for tuned ensembles, so Phase 1's 0.942 AUC is in the upper range and its 87.8% accuracy is well above it.

## 8.1 Related studies

| Study Data Method Reported result Evaluation protocol Smith et al., Pima, 768 rows ADAP, an early Sensitivity = 576 train / 192 1988 [1] neural network specificity = 76% test at cutoff 0.448 [2] Chang et Pima, 768 rows Naive Bayes, Random Forest Manual 70/30 al., 2022 [3] Random Forest, J48 accuracy 79.57%; split (538 / tree, in R, with an J48 sensitivity 230) interpretability 88.43%; Naive focus Bayes accuracy |
| --- |
| 79.13% on 3 features |


| Study | Data | Method | Reported result | Evaluation |
| --- | --- | --- | --- | --- |
|   |   |   |   | protocol |
| Hasan et | Pima, 768 rows Outlier rejection, |   | AUC 0.950, | K-fold cross- |
| al., 2020 [4] |   | imputation, | sensitivity 0.789, | validation, |
|   |   | standardisation, | specificity 0.934 | grid- |
|   |   | feature selection, |   | searched |
|   |   | then an AUC- |   |   |
|   |   | weighted ensemble |   |   |
|   |   | of KNN, DT, RF, |   |   |
|   |   | AdaBoost, NB, |   |   |
|   |   | XGBoost and MLP |   |   |
| Zou et al., | Luzhou hospital | Decision tree, | Random Forest | 5-fold cross- |
| 2018 [5] | data (main), | Random Forest, | accuracy 0.8084 | validation |
|   | Pima | neural network, | on the hospital |   |
|   | (secondary) | with PCA and | data with all |   |
|   |   | mRMR | attributes |   |
| Kavakiotis | Systematic | Machine learning | Survey of | Not |
| et al., 2017 | review | and data mining in | prediction, | applicable |
| [6] |   | diabetes research | complications and |   |
|   |   |   | biomarker work |   |
| This work | Merged Pima, | 7 classifiers, 8 | Accuracy 0.878, | Single 80/20 |
| (Phase 1) | 1,144 rows | selection | AUC 0.942; at | stratified |
|   |   | strategies, GA, | threshold 0.35 | hold-out |
|   |   | tuned Gradient | recall 0.938 | (229) |
|   |   | Boosting, threshold |   |   |
|   |   | 0.35 |   |   |

## 8.2 How Phase 1 compares

| Dimension | Prior work | Phase 1 |
| --- | --- | --- |
| Preprocessing | Hasan et al. also reject outliers, | KNN imputation fitted on train |
|   | impute and standardise [4] | only; no outlier handling |
| Feature selection | Chang et al. rank features; Zou et | Eight strategies across wrapper, |
|   | al. use PCA and mRMR [3, 5] | embedded and evolutionary |
|   |   | families, with a model-matched |
|   |   | GA |


| Dimension | Prior work | Phase 1 |
| --- | --- | --- |
| Representation | Zou et al. use PCA only [5] | 3 linear and 8 non-linear |
| analysis |   | projections |
| Model family | Chang: 3 interpretable models; | 7 models, one tuned; no |
|   | Hasan: 7 models plus ensemble | ensemble |
|   | [3, 4] |   |
| Decision rule | Fixed or implicit 0.5 cutoff; Smith | Explicit screening threshold |
|   | et al. set 0.448 [2] | study (0.30 to 0.50) |
| Validation | Cross-validation [4, 5] or a 70/30 | One hold-out split, no |
|   | split [3] | confidence intervals |
| Interpretability | Chang et al. make it a central goal | Selection overlap and L1 weights |
|   | [3] | only; no SHAP or partial |
|   |   | dependence |

## 8.3 Reading the numbers against the literature

- AUC is comparable. 0.942 for a single tuned Gradient Boosting model sits close to Hasan et al.'s 0.950 for a seven-model ensemble [4].

- Accuracy is not. 87.8% against 79.6% for Random Forest [3] is an 8-point gap on the same underlying patients, which is implausible for a modest model change and consistent with the leakage described below.

- Sensitivity is a design choice, not a gain. Recall of 0.938 at threshold 0.35 compares with Hasan et al.'s 0.789 only because the threshold was lowered; the cost is specificity of 0.811 (120 of 148 negatives).

- Leakage is a known failure mode. A survey of past reviews found leakage in at least 294 papers across 17 fields, and after correction complex ML models often did no better than older logistic-regression baselines [7]. Phase 1 shows the same signature: Logistic Regression at 0.85 AUC against 0.94 for boosting.

## 9. Critical assessment and Phase 2 plan

Two data issues, both traceable to merging the two Pima files, probably inflate every reported score, so Phase 2 must correct them before any model is compared again. The first two rows below are inferences from the notebook's printed outputs and need a quick check in code.


## 9.1 Validity issues

| # | Issue | Evidence in the | Effect | Fix |
| --- | --- | --- | --- | --- |
|   |   | notebook |   |   |
| 1 | Same patient in | 1,536 = 2 × 768; de- | Roughly half of test | Keep one row per |
|   | train and test | duplication leaves | rows (about 66% | patient, preferring |
|   |   | 1,144 = 392 | have a twin, 80% of | the raw file; split |
|   |   | identical rows + 2 | twins land in | by patient group |
|   |   | × 376 near- | training) have a |   |
|   |   | identical pairs | near-copy in |   |
|   |   | (same patient, | training; favours |   |
|   |   | one copy with | memorising models |   |
|   |   | filled values, one | such as trees and |   |
|   |   | with missing | KNN |   |
|   |   | values) |   |   |
| 2 | Label-derived | Insulin quartiles | Insulin acts as a | Use the raw file |
|   | imputation in | are 102.5 / 102.5 / | label proxy for | only; check with a |
|   | diabetes.csv | 169.5, and the first | about a third of | crosstab of Insulin |
|   |   | rows show | rows and would | against Outcome |
|   |   | Outcome 0 with | explain its selection |   |
|   |   | 102.5 and | by every method |   |
|   |   | Outcome 1 with |   |   |
|   |   | 169.5 |   |   |
| 3 | Test set used for | Models ranked by | Optimistic final | Choose models |
|   | selection | test ROC-AUC; | metrics | and threshold on |
|   |   | threshold 0.35 |   | out-of-fold |
|   |   | picked from a |   | predictions; touch |
|   |   | test-set table |   | the test set once |
| 4 | Single small split | 229 test patients, | Accuracy standard | Repeated |
|   |   | no intervals | error about 2.3 | stratified k-fold |
|   |   |   | points | plus bootstrap |
|   |   |   |   | confidence |
|   |   |   |   | intervals |


| # | Issue | Evidence in the | Effect | Fix |
| --- | --- | --- | --- | --- |
|   |   | notebook |   |   |
| 5 | Imputation outside | Imputer fitted | Mild leakage | Wrap imputer, |
|   | cross-validation | once on the whole | between CV folds | scaler and model |
|   |   | training set, then |   | in one Pipeline |
|   |   | reused inside |   |   |
|   |   | selection and |   |   |
|   |   | tuning CV |   |   |
| 6 | Narrow population Pima women aged |   | Limited | External |
|   |   | 21 or older from | generalisation | validation on |
|   |   | one region [2] |   | another cohort |
| 7 | Housekeeping | SVD duplicates | None on results | Drop SVD from |
|   |   | PCA; seaborn |   | the comparison; |
|   |   | palette |   | pass hue ; use |
|   |   | deprecation |   | relative paths |
|   |   | warnings; hard- |   |   |
|   |   | coded local file |   |   |
|   |   | paths |   |   |

## 9.2 Phase 2 recommendations

- Rebuild the dataset from the raw file only and report row counts at each step.

- Re-run the full benchmark with grouped, repeated cross-validation and report mean ± standard deviation.

- Re-run feature selection inside the cross-validation loop and compare the new subsets with Section 6.

- Select the screening threshold from out-of-fold predictions using an explicit cost ratio of a missed case to a false alarm.

- Add calibration (Brier score, reliability curve) and precision-recall AUC, which suit the 35% positive rate.

- Add a weighted ensemble and XGBoost-class baselines to match the strongest published comparator [4].

- Add SHAP or partial-dependence explanations to match the interpretability emphasis in [3].

- Add a trustworthiness score for the 2-D embeddings so the non-linear comparison is quantitative.


## 9.3 Reproducibility

Inputs are diabetes.csv and pima_indians_diabetes.csv ; the notebook writes

combined_diabetes.csv . All stochastic steps use seed 42. Library versions are not pinned; the printed warnings indicate Python 3.10 on Windows, so Phase 2 should add a

requirements.txt .

## 10. References

Sources [1]–[7] were opened during this write-up (2026-11-21); the method references [8]–[21] are standard citations given from bibliographic knowledge and were not re- opened.

## Dataset and related work

- 1. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proc. Annual Symposium on Computer Application in Medical Care, 261–265. PMC2245318 [URL 🔗](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318)

- 2. Pima Indians Diabetes Database documentation (NIDDK, donated by V. Sigillito, 1990): 768 instances, 8 attributes, ADAP results. pima.info [URL 🔗](https://www2.stat.duke.edu/courses/Fall00/sta290/datasets/pima.info)

- 3. Chang, V., Bailey, J., Xu, Q. A., Sun, Z. (2023). Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms. Neural Computing and Applications 35, 16157–16173. doi:10.1007/s00521-022-07049-z [URL 🔗](https://link.springer.com/10.1007/s00521-022-07049-z)

- 4. Hasan, M. K., Alam, M. A., Das, D., Hossain, E., Hasan, M. (2020). Diabetes prediction using ensembling of different machine learning classifiers. IEEE Access 8, 76516– 76531. doi:10.1109/ACCESS.2020.2989857 [URL 🔗](https://experts.boisestate.edu/en/publications/diabetes-prediction-using-ensembling-of-different-machine-learnin/)

- 5. Zou, Q., Qu, K., Luo, Y., Yin, D., Ju, Y., Tang, H. (2018). Predicting diabetes mellitus with machine learning techniques. Frontiers in Genetics 9, 515. doi:10.3389/fgene.2018.00515 [URL 🔗](https://pmc.ncbi.nlm.nih.gov/articles/6232260)

- 6. Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., Chouvarda, I. (2017). Machine learning and data mining methods in diabetes research. Computational and Structural Biotechnology Journal 15, 104–116. doi:10.1016/j.csbj.2016.12.005 [URL 🔗](https://pmc.ncbi.nlm.nih.gov/articles/PMC5257026)

- 7. Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in machine- learning-based science. Patterns 4(9), 100804. doi:10.1016/j.patter.2023.100804 [URL 🔗](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856)

## Methods

- 8. Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. Bioinformatics 17(6), 520–525.


- 9. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. Journal of Machine Learning Research 12, 2825–2830.

- 10. Schölkopf, B., Smola, A., Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. Neural Computation 10(5), 1299–1319.

- 11. Hinton, G., Roweis, S. (2002). Stochastic neighbor embedding. Advances in Neural Information Processing Systems 15.

- 12. van der Maaten, L., Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research 9, 2579–2605.

- 13. McInnes, L., Healy, J., Melville, J. (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.

- 14. Tenenbaum, J. B., de Silva, V., Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. Science 290, 2319–2323.

- 15. Roweis, S. T., Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. Science 290, 2323–2326.

- 16. Guyon, I., Weston, J., Barnhill, S., Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. Machine Learning 46, 389–422.

- 17. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society B 58(1), 267–288.

- 18. Zou, H., Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society B 67(2), 301–320.

- 19. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189–1232.

- 20. Breiman, L. (2001). Random forests. Machine Learning 45, 5–32.

- 21. Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.
