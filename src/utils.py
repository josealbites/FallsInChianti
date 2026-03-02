import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

# Modified by JAS 25.10.25

class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """ Custom transformer to remove highly correlated features to prevent multicollinearity. """
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.columns_to_keep_ = None
        self.indices_to_keep_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_indices = {column for column in upper.columns if any(upper[column] > self.threshold)}
        all_indices = set(range(X_df.shape[1]))
        self.indices_to_keep_ = sorted(list(all_indices - to_drop_indices))
        if hasattr(X, 'columns'):
            self.columns_to_keep_ = X.columns[self.indices_to_keep_]
        return self

    def transform(self, X, y=None):
        if self.indices_to_keep_ is None:
            raise RuntimeError("Transformer has not been fitted yet.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.indices_to_keep_]
        return X[:, self.indices_to_keep_]

def make_stratified_group_folds(y: pd.Series, groups: pd.Series, n_splits: int, random_state: int = 42):
    """ Creates folds stratified by outcome and grouped by participant ID to prevent data leakage. """
    rng = check_random_state(random_state)
    g_lbl = pd.DataFrame({'g': groups.values, 'y': y.values}).groupby('g', sort=False)['y'].max()
    uniq_groups, uniq_y = g_lbl.index.to_numpy(), g_lbl.to_numpy()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx, g_vals = np.arange(len(y)), groups.to_numpy()
    
    for g_tr, g_te in skf.split(uniq_groups, uniq_y):
        te_groups, tr_groups = uniq_groups[g_te], uniq_groups[g_tr]
        te_mask, tr_mask = np.isin(g_vals, te_groups), np.isin(g_vals, tr_groups)
        yield idx[tr_mask], idx[te_mask]
