import joblib
from sklearn.metrics import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class Ensemble:
    def __init__(self, method, X_train, X_val, y_train, y_val, num_rounds, sampling_name):
        self.method = method
        self.X_train = X_train
        self.X_test = X_val
        self.y_train = y_train
        self.y_test = y_val
        self.num_rounds = num_rounds
        self.name = sampling_name

    def save_dict_to_txt(self, file_path, dictionary):
        with open(file_path, 'w') as f:
            for key, value in dictionary.items():
                f.write(f'{key}: {value}\n')

    def DecisionTree(self, params):
        bst = DecisionTreeClassifier(**params)
        bst.fit(self.X_train, self.y_train)
        y_pred = bst.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = f1_score(self.y_test, predictions, average='micro')

        # joblib.dump(bst, f'results/dt_{self.name}_model.pkl')
        print("DT F1-Score:", accuracy)
        return accuracy

    def lightGBM(self, params):
        bst = LGBMClassifier(**params, n_estimators=self.num_rounds, verbose_eval=100)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
        y_pred = bst.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = f1_score(self.y_test, predictions, average='micro')

        # joblib.dump(bst, f'results/lgb_{self.name}_model.pkl')
        print("lightGBM F1-Score:", accuracy)
        return accuracy

    def XGBoost(self, params):
        bst = XGBClassifier(**params, n_estimators=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = f1_score(self.y_test, predictions, average='micro')

        # joblib.dump(bst, f'results/xgb_{self.name}_model.pkl')
        print("XGBoost F1-Score:", accuracy)
        return accuracy

    def CatBoost(self, params):
        bst = CatBoostClassifier(**params, iterations=self.num_rounds)
        bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
        y_pred = bst.predict_proba(self.X_test)
        predictions = y_pred.argmax(axis=1)
        accuracy = f1_score(self.y_test, predictions, average='micro')

        # joblib.dump(bst, f'results/cat_{self.name}_model.pkl')
        print("CatBoost F1-Score:", accuracy)
        return accuracy

    def objective(self, trial):
        if self.method == 0:
            params = {
                'criterion': 'entropy',
                'max_features': trial.suggest_float('max_features', 0.1, 0.9),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
            }
            accuracy = self.DecisionTree(params)

        if self.method == 1:
            params = {
                'device': 'cpu',
                'num_class': 7,
                'early_stopping_rounds': 10,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbrt',
                'tree_learner': 'voting',

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                # 'num_leaves': trial.suggest_int('num_leaves', 50, 300),
            }
            accuracy = self.lightGBM(params)

        if self.method == 2:
            params = {
                'device': 'cuda',
                'num_class': 7,
                'early_stopping_rounds': 10,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',

                'eta': trial.suggest_float('eta', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            accuracy = self.XGBoost(params)

        if self.method == 3:
            params = {
                'task_type': 'GPU',
                'classes_count': 7,
                'early_stopping_rounds': 10,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'grow_policy': 'Lossguide',

                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'depth': trial.suggest_int('depth', 5, 16),
            }
            accuracy = self.CatBoost(params)

        return accuracy

    def save_best_model(self, best_params):
        if self.method == 0:
            best_params.update({
                'criterion': 'entropy',
            })

            bst = DecisionTreeClassifier(**best_params)
            bst.fit(self.X_train, self.y_train)
            joblib.dump(bst, f'results/dt_{self.name}_model.pkl')
            self.save_dict_to_txt(f'results/dt_{self.name}_params.txt', best_params)
            print(f'{f1_score(self.y_test, bst.predict(self.X_test), average="micro"):.4f}')
            print("Model saved!")

        if self.method == 1:
            best_params.update({
                'device': 'cpu',
                'num_class': 7,
                'early_stopping_rounds': 10,
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'boosting_type': 'gbrt',
                'tree_learner': 'voting',
            })

            bst = LGBMClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)])
            joblib.dump(bst, f'results/lgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'results/lgb_{self.name}_params.txt', best_params)
            print(f'{f1_score(self.y_test, bst.predict(self.X_test), average="micro"):.4f}')
            print("Model saved!")

        if self.method == 2:
            best_params.update({
                'device': 'cuda',
                'num_class': 7,
                'early_stopping_rounds': 10,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
            })

            bst = XGBClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'results/xgb_{self.name}_model.pkl')
            self.save_dict_to_txt(f'results/xgb_{self.name}_params.txt', best_params)
            print(f'{f1_score(self.y_test, bst.predict(self.X_test), average="micro"):.4f}')
            print("Model saved!")

        if self.method == 3:
            best_params.update({
                'task_type': 'GPU',
                'classes_count': 7,
                'early_stopping_rounds': 10,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'grow_policy': 'Lossguide',
                'bootstrap_type': 'Bayesian',
            })

            bst = CatBoostClassifier(**best_params)
            bst.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=100)
            joblib.dump(bst, f'results/cat_{self.name}_model.pkl')
            self.save_dict_to_txt(f'results/cat_{self.name}_params.txt', best_params)
            print(f'{f1_score(self.y_test, bst.predict(self.X_test), average="micro"):.4f}')
            print("Model saved!")
