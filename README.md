# Customer Loan Rate Classification (tabular competition project, 2024)

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Gradient boosting](https://img.shields.io/badge/LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-ensemble-2ea043)](src/ensemble.py)
[![Optuna](https://img.shields.io/badge/Optuna-100%20trials%20per%20model-7B61FF)](main.py)
[![Task](https://img.shields.io/badge/task-7--class%20loan%20grade-FFB000)](data/train.csv)

Predict the credit grade a lender assigned to a consumer loan (`대출등급`, seven classes A to G)
from the loan application and the borrower's credit record. The data is a Korean tabular
competition release: 96,294 labelled rows with 13 raw features and 64,197 unlabelled test rows,
provided as `train.csv` / `test.csv` / `sample_submission.csv`. The grades are heavily
imbalanced — B 29.9%, C 28.7%, A 17.4%, D 13.9%, E 7.6%, F 2.0%, G 0.4%.

The approach is a feature-engineering problem rather than a modelling one. Two ratios derived
from the repayment columns, `총상환이자 / 대출금액` (interest paid over principal borrowed) and
`총상환원금 / 대출금액`, carry almost all of the signal; the raw repayment columns are then dropped.
On top of that, four tree learners (decision tree, LightGBM, XGBoost, CatBoost) are tuned
independently with Optuna and combined by soft voting. Because the rare grades F and G are only
2.4% of the data together, the project compares seven ways of rebalancing the training set,
including a partial oversampler written for this repo that lifts a minority class only up to
30% of the majority instead of all the way to parity.

## Pipeline

```
data/train.csv (96,294 x 15)
  │  drop ID, 80/20 holdout split (seed 42)          -> 77,035 fit / 19,259 holdout
  ▼
preprocessing.ipynb
  │  bin           부채_대비_소득_비율 -> 5 bins, 총연체금액 -> 2, 연체계좌수 -> 3,
  │                최근_2년간_연체_횟수 -> 4 bins
  │  parse         근로기간 "10+ years" -> 10 (regex digits, NaN -> 0)
  │  encode        LabelEncoder on the 7 discrete columns and on the target
  │  derive        이자/대출 = 총상환이자/대출금액 , 원금/대출 = 총상환원금/대출금액
  │  winsorize     연간소득, 0.05% on each tail
  │  drop          총상환원금, 총상환이자  (now represented by the two ratios)
  │  split         80/20 again (seed 42)              -> 61,628 train / 15,407 val
  │  scale         StandardScaler on the 6 continuous columns, fitted on train only
  ▼
data/X_train.csv X_val.csv y_train.csv y_val.csv (13 features)  +  data/scaler.joblib
  ▼
main.py  --  for each resampler in {None, SMOTE, ADASYN, KMeansSMOTE,
  │          SMOTE_all, KMeansSMOTE_all, Custom_SMOTE}
  │          for each learner in {DecisionTree, LightGBM, XGBoost, CatBoost}
  │            Optuna, 100 trials, maximise micro-F1 on X_val
  │            refit on the best trial, early stopping 10, 1000 rounds
  ▼
results/{dt,lgb,xgb,cat}_<resampler>_model.pkl  +  _params.txt
  ▼
postprocessing.ipynb
  │  predict_proba of the 4 models on the 19,259-row holdout
  │  soft voting = mean of the four probability matrices, argmax
  ▼
results/metric_<resampler>.csv  +  results/images/output.png (feature importances)
```

`eda.ipynb` and `eda_2.ipynb` sit beside this path and only read data. The first inspects the
raw CSV with Korean labels; the second reads the preprocessed matrix, inverts the scaler to
recover the original units, and relabels everything in English to produce the report figures.

## Results

Weighted-average metrics on the 19,259-row labelled holdout (the 20% of `train.csv` never seen
during fitting or tuning), as stored in `results/metric_*.csv`. Accuracy and weighted recall are
the same quantity, so recall is omitted. Soft voting is the mean of the four probability
matrices.

### Soft voting, by resampling strategy

| Training set | Accuracy | Precision | F1 | AUC (OvR) |
|---|---|---|---|---|
| **Custom_SMOTE** | **0.94823** | 0.94860 | **0.94824** | **0.99750** |
| None (no resampling) | 0.94776 | 0.94809 | 0.94769 | 0.99742 |
| SMOTE (minority) | 0.94756 | **0.94967** | 0.94792 | 0.99740 |
| ADASYN (minority) | 0.94512 | 0.94685 | 0.94551 | 0.99736 |
| KMeansSMOTE (minority) | 0.94449 | 0.94541 | 0.94469 | 0.99730 |
| KMeansSMOTE (all classes) | 0.94133 | 0.94208 | 0.94157 | 0.99677 |
| SMOTE (all classes) | 0.94107 | 0.94178 | 0.94128 | 0.99682 |

Rebalancing every class to parity (`*_all`) is the worst option here, and full oversampling of
the single smallest class is roughly a wash against doing nothing. The partial oversampler of
`src/custom_smote.py` is the only variant above the no-resampling baseline on all four metrics.

### Per model, on the Custom_SMOTE training set

| Model | Accuracy | Precision | F1 | AUC (OvR) |
|---|---|---|---|---|
| Decision tree | 0.94797 | 0.94870 | 0.94811 | 0.98777 |
| LightGBM | 0.94252 | 0.94280 | 0.94249 | 0.99713 |
| XGBoost | 0.94200 | 0.94231 | 0.94202 | 0.99706 |
| CatBoost | 0.93094 | 0.93128 | 0.93090 | 0.99575 |
| Soft voting | **0.94823** | **0.94860** | **0.94824** | **0.99750** |

A single tuned decision tree is within 0.0002 F1 of the four-model ensemble, which is what the
feature importances say as well: `원금/대출` and `이자/대출` dominate every model
(`results/images/output.png`). The gradient boosters pay for their lower accuracy with much
better-calibrated probabilities, and the ensemble inherits the best AUC of the group.

The sanity-check tree inside `preprocessing.ipynb` (`max_depth=52`) reaches a **micro**-F1 of
0.9457 on the validation split and 0.9440 on the holdout; that number is on a different metric
and is not comparable with the tables above.

No competition leaderboard score is recorded anywhere in this repository, and none is claimed.

## Repository layout

| Path | Content |
|---|---|
| `preprocessing.ipynb` | train/holdout split, binning, encoding, the two derived ratios, scaling; writes `data/X_*.csv`, `data/y_*.csv`, `data/scaler.joblib` |
| `main.py` | the training driver: picks a resampler, then runs Optuna over the four learners |
| `postprocessing.ipynb` | loads the saved models, soft voting, metric table, feature-importance figure |
| `eda.ipynb`, `eda_2.ipynb` | distributions, class balance, grade-vs-feature plots; `eda.ipynb` on the raw CSV with Korean labels, `eda_2.ipynb` on the preprocessed matrix with English labels |
| `src/ensemble.py` | `Ensemble`: the Optuna objective and the refit-and-save step for all four learners |
| `src/custom_smote.py` | `CustomSMOTE`: partial oversampling to 30% of the majority class, K=5 neighbours, interpolating only the six continuous columns |
| `data/` | `train.csv`, `test.csv`, `sample_submission.csv`, the preprocessed matrices and the fitted scaler |
| `results/` | one `.pkl` model and one `_params.txt` per (learner, resampler), `metric_*.csv`, `total.csv`, `images/output.png` |
| `reference/` | archival: an earlier ABC/DEFG two-stage attempt, a CatBoost tutorial, a paper PDF, and code from an unrelated temperature-regression project. Not part of the pipeline; its relative paths are stale |
| `ARCHITECTURE.md` | module map, coupling, complexity ranking |

## Get Started

**1. Install.** The code was last run on CPython 3.11. Versions are not pinned because the
original environment was not recorded.

```bash
conda create -n loan python=3.11
conda activate loan
pip install -r requirements.txt
```

XGBoost is configured with `device='cuda'` and CatBoost with `task_type='GPU'` in
`src/ensemble.py`. Without a GPU, methods 2 and 3 of `main.py` will not run as written.

**2. Data.** `data/` already contains the competition CSVs and the preprocessed matrices, so the
pipeline can be entered at any stage. To rebuild the matrices from `data/train.csv`, run
`preprocessing.ipynb` from the repository root and uncomment its two `to_csv` cells (they are
commented out so that a casual run does not overwrite the stored inputs).

**3. Train.** Every notebook and script expects the repository root as the working directory.

```bash
jupyter nbconvert --to notebook --execute preprocessing.ipynb   # optional, see above
python main.py                                                  # writes results/*.pkl
jupyter nbconvert --to notebook --execute postprocessing.ipynb  # writes results/metric_*.csv
```

`main.py` runs the resamplers listed in its `sampling_name` list — currently
`['Custom_SMOTE']`. Add the other names (`'None'`, `'SMOTE'`, `'ADASYN'`, `'KMeanSMOTE'`,
`'SMOTE_all'`, `'KMeanSMOTE_all'`) to reproduce the full table; the same names are the
`name_list` of `postprocessing.ipynb`, and the two lists have to agree because the stages are
linked by filename.

**4. Inspect a single resampler.** `src/custom_smote.py` has a `__main__` block that prints the
class distribution before and after both the custom oversampler and plain SMOTE:

```bash
cd src && python custom_smote.py
```
