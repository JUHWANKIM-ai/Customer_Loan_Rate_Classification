import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

class CustomSMOTE:
    def __init__(self):
        pass

    def nearest_neighbour(self, X, x, K, features):
        X_selected = X.iloc[:, features]
        x_selected = x[features]

        distances = np.linalg.norm(X_selected - x_selected, axis=1)
        indices = np.argsort(distances)[1:K + 1]

        differences = X_selected.iloc[indices] - x_selected
        weights = np.random.random((K, 1))
        additive = differences * weights
        return additive

    def generate_new_samples(self, row, X, K, features):
        row = row.values
        additive = self.nearest_neighbour(X, row, K, features)
        new_samples = np.tile(row, (K, 1))
        new_samples[:, features] += additive
        return new_samples

    def integrate_result(self, X, y, to_generated_num, K, features):
        new_samples_list = X.apply(lambda row: self.generate_new_samples(row, X, K, features), axis=1)
        new_samples = np.vstack(new_samples_list)

        if new_samples.shape[0] < to_generated_num:
            to_generated_num = new_samples.shape[0]

        random_idx = np.random.choice(new_samples.shape[0], size=to_generated_num, replace=False)
        random_samples = new_samples[random_idx, :]

        self.X = np.vstack([self.X, random_samples])
        self.y = np.hstack([self.y, np.repeat(y.unique(), to_generated_num)])

    def SMOTE(self, X, y, K, features=None):
        if features is None:
            features = list(range(X.shape[1]))

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        self.X = X.values
        self.y = y.values

        unique_class = list(y.unique())

        class_indices = []
        for i in range(len(unique_class)):
            indices = y[y == unique_class[i]].index.tolist()
            class_indices.append(indices)

        class_amounts = []
        for i in range(len(class_indices)):
            class_amounts.append(len(class_indices[i]))

        class_proportions = []
        for i in range(len(class_amounts)):
            class_proportions.append(class_amounts[i] / max(class_amounts))

        to_smote_idx_list = []
        to_smote_amount_list = []
        for i in range(len(class_proportions)):
            if class_proportions[i] < 0.3:
                to_smote_idx_list.append(class_indices[i])
                num = int(max(class_amounts) * 0.3) - class_amounts[i]
                to_smote_amount_list.append(num)

        for i in range(len(to_smote_idx_list)):
            self.integrate_result(X.loc[to_smote_idx_list[i]], y[to_smote_idx_list[i]], to_smote_amount_list[i], K,
                                  features)

        return self.X, self.y


if __name__ == "__main__":
    X_train = pd.read_csv('../data/X_train.csv', index_col=0)
    y_train = pd.read_csv('../data/y_train.csv', index_col=0)
    # SMOTE 적용
    features = [0, 2, 4, 6, 11, 12]
    S = CustomSMOTE()
    X_balanced, y_balanced = S.SMOTE(X_train, y_train, 5, features)

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("Original class distribution:", Counter(y_train.iloc[:, 0].values))
    print("CUSTOM SMOTE distribution:", Counter(y_balanced))
    print(X_balanced.shape)
    print(y_balanced.shape)
    print("SMOTE distribution:", Counter(y_res.iloc[:, 0].values))
    print(X_res.shape)
    print(y_res.shape)


