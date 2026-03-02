import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import pickle

from utils import DropCorrelatedFeatures, make_stratified_group_folds

def build_model_spaces(numeric_feats: List[str], categorical_feats: List[str]):
    """ Defines the preprocessing pipeline and hyperparameter spaces. """
