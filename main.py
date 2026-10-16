import optuna
import warnings
import pandas as pd
from src.ensemble import *
from src.custom_smote import CustomSMOTE
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    X_train = pd.read_csv('data/X_train.csv', index_col=0)
    y_train = pd.read_csv('data/y_train.csv', index_col=0)
    X_val = pd.read_csv('data/X_val.csv', index_col=0)
    y_val = pd.read_csv('data/y_val.csv', index_col=0)

    sampling_name = ['Custom_SMOTE']
    for i, name in enumerate(sampling_name):
        if name == 'SMOTE':
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        elif name == 'ADASYN':
            adasyn = ADASYN(sampling_strategy='minority', random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train, y_train)

        elif name == 'KMeanSMOTE':
            kmsmote = KMeansSMOTE(sampling_strategy='minority', random_state=42, kmeans_estimator=7,
                                  cluster_balance_threshold=0.01)
            X_train, y_train = kmsmote.fit_resample(X_train, y_train)

        elif name == 'SMOTE_all':
            smote = SMOTE(sampling_strategy='all', random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        elif name == 'KMeanSMOTE_all':
            kmsmote = KMeansSMOTE(sampling_strategy='all', random_state=42, kmeans_estimator=7,
                                  cluster_balance_threshold=0.01)
            X_train, y_train = kmsmote.fit_resample(X_train, y_train)

        elif name == 'Custom_SMOTE':
            features = [0, 2, 4, 6, 11, 12]
            S = CustomSMOTE()
            X_train, y_train = S.SMOTE(X_train, y_train, 5, features)
            X_val = X_val.values
            y_val = y_val.values

        for i in range(4):
            # {DT=0, lightGBM=1, XGBoost=2, CatBoost=3}
            E = Ensemble(i, X_train, X_val, y_train, y_val, 1000, name)

            study = optuna.create_study(direction='maximize')
            study.optimize(E.objective, n_trials=100)

            print('Number of finished trials:', len(study.trials))
            print('Best trial:', study.best_trial.params)
            E.save_best_model(study.best_trial.params)
