# Architecture

A map of the repository: what runs, what each part owns, and where the coupling is.
Written from the code and the stored notebook outputs; nothing here was re-executed
(the pipeline needs a GPU for the XGBoost and CatBoost arms).

## Structure

```
.
├── eda.ipynb              # raw-CSV inspection (Korean labels, GULIM font)
├── eda_2.ipynb            # preprocessed matrix, scaler inverted, English labels, saves figures
├── preprocessing.ipynb    # train.csv -> data/X_train.csv, X_val.csv, y_*.csv, scaler.joblib
├── main.py                # Optuna search over 4 learners x 1 resampler -> results/*.pkl
├── postprocessing.ipynb   # loads results/*.pkl, soft voting, results/metric_*.csv
├── src/
│   ├── ensemble.py        # class Ensemble: 4 learners behind one objective()
│   └── custom_smote.py    # class CustomSMOTE: partial oversampling to 30% of the majority
├── data/                  # competition CSVs + the preprocessed matrices + the fitted scaler
├── results/               # one .pkl model and one .txt parameter dump per (learner, resampler)
│                          # + metric_<resampler>.csv, total.csv, images/output.png
└── reference/             # archival: earlier attempts and material from another project
```

Entry points, in pipeline order: `preprocessing.ipynb` -> `main.py` -> `postprocessing.ipynb`.
The two EDA notebooks are read-only side branches.

Every entry point is executed **from the repository root**; all data paths inside them are
relative to it (`data/...`, `results/...`). The one exception is the `__main__` block of
`src/custom_smote.py`, a self-test that is run from inside `src/` and reads `../data/`.

## Dependencies

Internal fan-in (who is imported by whom):

| Module | Imported by | Public surface |
|---|---|---|
| `src/ensemble.py` | `main.py` | `Ensemble(method, X_train, X_val, y_train, y_val, num_rounds, sampling_name)`, `.objective(trial)`, `.save_best_model(best_params)` |
| `src/custom_smote.py` | `main.py`, its own `__main__` | `CustomSMOTE().SMOTE(X, y, K, features)` |

`src/ensemble.py` is the only module with fan-in from more than its own file and is the
de-facto interface of the project: `main.py` never names a learner, it passes an integer
(`0=DecisionTree, 1=LightGBM, 2=XGBoost, 3=CatBoost`) and lets `Ensemble` decide the search
space, the fit and the metric. The notebooks do not import either module; they talk to the
pipeline only through files (`data/*.csv`, `results/*.pkl`).

External: scikit-learn, LightGBM, XGBoost, CatBoost, imbalanced-learn, Optuna, pandas, numpy,
scipy, joblib, matplotlib, seaborn (see `requirements.txt`). `reference/` additionally uses
torch, statsmodels and arch; those are deliberately left out of `requirements.txt` because no
code on the live path needs them.

The contract between the stages is the filename, not a function signature:
`Ensemble.save_best_model` writes `results/{dt,lgb,xgb,cat}_{sampling_name}_model.pkl`, and
`postprocessing.ipynb` rebuilds exactly those names from `name_list[i]`. Renaming a resampler
in `main.py` silently breaks the notebook; this is the sharpest coupling in the repo.

## Conventions

- Entry points are notebooks or `main.py` at the root; reusable code lives in `src/`.
- Files and directories are lower_snake_case; classes are CapWords.
- Data in `data/`, model and metric artifacts in `results/`, never mixed.
- Column names in the data stay Korean (they come from the competition file); figures map them
  to English through the dictionary repeated in `eda_2.ipynb` and `postprocessing.ipynb`.
- Random seeds are fixed at 42 for every split and every resampler.
- The score used for model selection is `f1_score(..., average='micro')` on the validation
  split; the score reported in `results/metric_*.csv` is `average='weighted'` on the holdout.
  They are different metrics on different sets — do not compare them directly.

## Impact

Where a change hurts, highest first.

| Risk | Where | Why |
|---|---|---|
| Resampler names | `main.py` `sampling_name`, `postprocessing.ipynb` `name_list` | the two lists must agree; the link is a filename, not an import |
| `Ensemble.save_best_model` | `src/ensemble.py:124-188` | four near-identical branches, each repeating the parameter dict of `objective()`; a parameter added in one place and not the other changes the saved model without any error |
| `Ensemble.objective` | `src/ensemble.py:68-122` | same four-branch shape; `if` rather than `elif`, so `accuracy` is only defined because the branches are disjoint by construction |
| Preprocessing bin edges | `preprocessing.ipynb` | the bins are hard-coded twice, once for train and once for test; they must stay identical or the encodings diverge |
| `CustomSMOTE.SMOTE` | `src/custom_smote.py:42-79` | the most branching function on the live path; it mutates `self.X`/`self.y` inside `integrate_result`, so the object is not reusable across calls |
| Device flags | `src/ensemble.py` | XGBoost is pinned to `device='cuda'` and CatBoost to `task_type='GPU'`; without a GPU `main.py` fails on methods 2 and 3 |

**Complexity ranking** (lizard, live path only; CRAP_max = cc² + cc, the score a function would
have with zero test coverage):

| Function | NLOC | CC | CRAP_max |
|---|---|---|---|
| `CustomSMOTE.SMOTE` | 29 | 9 | 90 |
| `Ensemble.objective` | 46 | 5 | 30 |
| `Ensemble.save_best_model` | 58 | 5 | 30 |
| `CustomSMOTE.integrate_result` | 9 | 2 | 6 |

The repository has no tests, so the real CRAP score cannot be computed; the column above is the
worst case (coverage = 0) and is an upper bound used only to rank the functions.

## What was deliberately left alone

`reference/` is archival. It holds an earlier two-stage attempt at this task
(`base.ipynb`, `Parameter_Tuning.ipynb`, `Postprocessing.ipynb`, which split the grades into
ABC/DEFG), files from a different project of the same author (`0_preprocessing.py`,
`1_main.py`, `Transfer_Learning.py`, `EDA.ipynb`, all about temperature regression), a CatBoost
tutorial notebook and a paper PDF. Its paths still point at `../Database/` and at CSVs that are
not in this repository; they were left untouched rather than rewritten to a layout that would
still not run.
