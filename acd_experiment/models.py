from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDOneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# from imblearn.over_sampling import SMOTENC
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn import FunctionSampler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

import shap

from typing import Dict, Any, Iterable, Optional

from .utils import group_explanations_by_categorical

from joblib import Parallel, delayed

class Estimator:
    _name: str
    _estimator: BaseEstimator
    _requirements: Dict[str, bool]
    _static_params: Dict[str, Any] = {}
    _tuning_params_default: Dict[str, Any] = {}
    _fit_params: Dict[str, Any] = {}
    _explainer_args: Dict[str, Any] = {}

    def __init__(self, train):
        pass

    @classmethod
    def suggest_parameters(cls, trial):
        return dict()

    @classmethod
    def compile_parameters(cls, params):
        return {
            f"{cls._name}__{key}": value
            for key, value in {
                **cls._static_params,
                **cls._tuning_params_default,
                **params,
            }.items()
        }

    @classmethod
    def factory(cls, **kwargs):
        return cls._estimator(**(cls._static_params | kwargs))

    @classmethod
    def fit_params(cls, X_train, y_train):
        return {f"{cls._name}__{key}": value for key, value in cls._fit_params.items()}

    @classmethod
    def explain_calibrated(cls, model, X_train, X_test, cv_jobs=1):
        # ordinal_encode = (
        #     not cls._requirements["onehot"] and not cls._requirements["ordinal"]
        # )
        X = X_test#.ordinal_encode_categories() if ordinal_encode else X_test
        if cls._requirements["scaling"]:
            X_trains, X_tests = (
                [
                    _.base_estimator["Scaling"].fit_transform(X_train)
                    for _ in model.calibrated_classifiers_
                ],
                [
                    _.base_estimator["Scaling"].fit_transform(X)
                    for _ in model.calibrated_classifiers_
                ],
            )
        else:
            X_trains, X_tests = (
                [X_train for _ in model.calibrated_classifiers_],
                [X for _ in model.calibrated_classifiers_],
            )

        if cv_jobs > 1:
            explainers = Parallel(n_jobs=cv_jobs, verbose=10)(
                delayed(
                    cls._explainer(
                        _.base_estimator[cls._name],
                        masker=X_trains[i],
                        **cls._explainer_args,
                    )
                )(X_tests[i])
                for i, _ in enumerate(model.calibrated_classifiers_)
            )
        else:
            explainers = [
                cls._explainer(
                    _.base_estimator[cls._name],
                    masker=X_trains[i],
                    **cls._explainer_args,
                )(X_tests[i])
                for i, _ in enumerate(model.calibrated_classifiers_)
            ]

        shap_values = shap.Explanation(
            base_values=np.array([_.base_values for _ in explainers]).mean(axis=0),
            values=np.array([_.values for _ in explainers]).mean(axis=0),
            data=X_test.values,
            feature_names=X_test.columns,
        )

        if cls._requirements["onehot"]:
            cols = X_test.get_onehot_categorical_columns()
            if len(cols):
                shap_values = group_explanations_by_categorical(
                    shap_values, X_test, cols
                )

        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]

        return shap_values

    @classmethod
    def explain(cls, model, X_train, X_test):
        # ordinal_encode = (not cls._requirements["onehot"]) and (
        #     not cls._requirements["ordinal"]
        # )

        shap_values = cls._explainer(model, X_train, **cls._explainer_args)(
            X_test#.ordinal_encode_categories() if ordinal_encode else X_test
        )

        if cls._requirements["onehot"]:
            cols = X_test.get_onehot_categorical_columns()
            if len(cols) and False:
                shap_values = group_explanations_by_categorical(
                    shap_values, X_test, cols
                )
        elif ordinal_encode:
            shap_values = shap.Explanation(
                base_values=shap_values.base_values,
                values=shap_values.values,
                data=X_test.values,
                feature_names=X_test.columns,
            )

        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]

        return shap_values


class Estimator_XGBoost(Estimator):
    _name = "XGBoost"
    _estimator = XGBClassifier

    _requirements = dict(
        onehot=False,
        ordinal=True,
        imputation=False,
        fillna=False,
        oneclass=False,
        calibration=True,
        explanation=False,
        scaling=False,
    )

    _static_params = dict(
        verbosity=0,
        n_jobs=1,
        objective="binary:logistic",
        booster="gbtree",
        enable_categorical=False,
    )

    _tuning_params_default = dict(
        tree_method="hist",
        alpha=7e-05,
        subsample=0.42,
        colsample_bytree=0.87,
        scale_pos_weight=14,
        max_depth=7,
        min_child_weight=10,
        eta=0.035,
        gamma=4e-08,
        grow_policy="lossguide",
    ) | {"lambda": 7e-2}

    _explainer = shap.TreeExplainer

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            tree_method=trial.suggest_categorical(
                f"{cls._name}__tree_method", ["approx", "hist"]
            ),
            alpha=trial.suggest_float(f"{cls._name}__alpha", 1e-8, 1.0, log=True),
            subsample=trial.suggest_float(f"{cls._name}__subsample", 0.2, 1.0),
            colsample_bytree=trial.suggest_float(
                f"{cls._name}__colsample_bytree", 0.2, 1.0
            ),
            scale_pos_weight=trial.suggest_int(
                f"{cls._name}__scale_pos_weight", 1, 100
            ),
            max_depth=trial.suggest_int(f"{cls._name}__max_depth", 3, 9, step=2),
            min_child_weight=trial.suggest_int(f"{cls._name}__min_child_weight", 2, 10),
            eta=trial.suggest_float(f"{cls._name}__eta", 1e-8, 1.0, log=True),
            gamma=trial.suggest_float(f"{cls._name}__gamma", 1e-8, 1.0, log=True),
            grow_policy=trial.suggest_categorical(
                f"{cls._name}__grow_policy", ["depthwise", "lossguide"]
            ),
        )
        suggestions["lambda"] = trial.suggest_float(
            f"{cls._name}__lambda", 1e-8, 1.0, log=True
        )

        return cls.compile_parameters(suggestions)


class Estimator_LightGBM(Estimator):
    _name = "LightGBM"
    _estimator = LGBMClassifier

    _requirements = dict(
        onehot=False,
        ordinal=False,
        imputation=False,
        fillna=False,
        oneclass=False,
        calibration=True,
        explanation=True,
        scaling=False,
    )

    _static_params = dict(
        objective="binary",
        metric=["l2", "auc"],
        boosting_type="gbdt",
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )

    _tuning_params_default = dict(
        is_unbalance=True,
        reg_alpha=1.8e-3,
        reg_lambda=6e-4,
        num_leaves=14,
        colsample_bytree=0.4,
        subsample=0.97,
        subsample_freq=1,
        min_child_samples=6,
    )

    _explainer = shap.TreeExplainer

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            reg_alpha=trial.suggest_float(
                f"{cls._name}__reg_alpha", 1e-4, 10.0, log=True
            ),
            reg_lambda=trial.suggest_float(
                f"{cls._name}__reg_lambda", 1e-4, 10.0, log=True
            ),
            num_leaves=trial.suggest_int(f"{cls._name}__num_leaves", 2, 256),
            colsample_bytree=trial.suggest_float(
                f"{cls._name}__colsample_bytree", 0.4, 1.0
            ),
            subsample=trial.suggest_float(f"{cls._name}__subsample", 0.4, 1.0),
            subsample_freq=trial.suggest_int(f"{cls._name}__subsample_freq", 1, 7),
            min_child_samples=trial.suggest_int(
                f"{cls._name}__min_child_samples", 5, 150
            ),
            is_unbalance=trial.suggest_categorical(
                f"{cls._name}__is_unbalance", [True, False]
            ),
        )

        if not suggestions["is_unbalance"]:
            suggestions["scale_pos_weight"] = trial.suggest_int(
                f"{cls._name}__scale_pos_weight", 1, 100
            )

        r = cls.compile_parameters(suggestions)
        if not suggestions["is_unbalance"]:
            del r[f"{cls._name}__is_unbalance"]

        return r


class Estimator_LogisticRegression(Estimator):
    _name = "LogisticRegression"
    _estimator = LogisticRegression

    _requirements = dict(
        onehot=True,
        ordinal=False,
        imputation=True,
        fillna=True,
        oneclass=False,
        calibration=True,
        explanation=True,
        scaling=True,
    )

    _static_params = dict(
        max_iter=2000, solver="lbfgs", random_state=42, penalty="none"
    )

    _tuning_params_default = dict(C=5.9, class_weight="balanced")

    _explainer = shap.LinearExplainer
    _explainer_args = dict(feature_perturbation="correlation_dependent")

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            # penalty=trial.suggest_categorical(f"{cls._name}__penalty", ["l2", "none"]),
            #  C=trial.suggest_float(f"{cls._name}__C", 0.01, 10),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced"]
            ),
        )

        return cls.compile_parameters(suggestions)


class Estimator_L1Regression(Estimator_LogisticRegression):
    _name = "L1Regression"
    _static_params = dict(max_iter=2000, solver="saga", random_state=42, penalty="l1")

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            C=trial.suggest_float(f"{cls._name}__C", 0.01, 10),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced"]
            ),
        )

        return cls.compile_parameters(suggestions)


class Estimator_ElasticNetRegression(Estimator_LogisticRegression):
    _name = "ElasticNetRegression"
    _static_params = dict(
        max_iter=2000, solver="saga", random_state=42, penalty="elasticnet"
    )
    _tuning_params_default = dict(C=5.9, l1_ratio=0.5, class_weight="balanced")

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            C=trial.suggest_float(f"{cls._name}__C", 0.01, 10),
            l1_ratio=trial.suggest_float(f"{cls._name}__l1_ratio", 0.01, 0.99),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced"]
            ),
        )

        return cls.compile_parameters(suggestions)


class Estimator_L2Regression(Estimator_LogisticRegression):
    _name = "L2Regression"
    _static_params = dict(max_iter=2000, solver="lbfgs", random_state=42, penalty="l2")

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            C=trial.suggest_float(f"{cls._name}__C", 0.01, 10),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced"]
            ),
        )

        return cls.compile_parameters(suggestions)


class Estimator_RandomForest(Estimator):
    _estimator = RandomForestClassifier
    _name = "RandomForest"

    _requirements = dict(
        onehot=False,
       # ordinal=True,
        ordinal=False,
        imputation=False,
        fillna=True,
        oneclass=False,
        calibration=True,
        explanation=True,
        scaling=False,
    )
    _tuning_params_default = dict(
        n_estimators=250,
        max_features=0.56,
        min_samples_split=8,
        min_samples_leaf=3,
        max_samples=0.75,
        class_weight="balanced",
    )

    _explainer = shap.TreeExplainer

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            n_estimators=trial.suggest_int(f"{cls._name}__n_estimators", 25, 250),
            max_features=trial.suggest_float(f"{cls._name}__max_features", 0.15, 1.0),
            min_samples_split=trial.suggest_int(
                f"{cls._name}__min_samples_split", 2, 15
            ),
            min_samples_leaf=trial.suggest_int(f"{cls._name}__min_samples_leaf", 1, 15),
            max_samples=trial.suggest_float(f"{cls._name}__max_samples", 0.5, 0.99),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced", "balanced_subsample"]
            ),
        )

        return cls.compile_parameters(suggestions)


class IsolationForestWrapper(IsolationForest):
    """ Wraps the scikit-learn Isolation Forest model to adapt it to our test harness. This is because the original gives non-standard outputs when predicting """

    def predict(self, X):
        return np.fromiter(map({-1: 0, 1: 1}.get, super().predict(X)), dtype=int)

    def decision_function(self, X):
        return -super().decision_function(X)


class OneClassSVMWrapper(SGDOneClassSVM):
    def predict(self, X):
        return np.fromiter(map({-1: 0, 1: 1}.get, super().predict(X)), dtype=int)

    def decision_function(self, X):
        return super().decision_function(X)


class Estimator_IsolationForest(Estimator):
    _name = "IsolationForest"
    _estimator = IsolationForestWrapper
    _requirements = dict(
        onehot=True,
        ordinal=False,
        imputation=True,
        fillna=True,
        oneclass=True,
        calibration=False,
        explanation=False,
        scaling=False,
    )

    _static_params = dict(contamination="auto")

    _tuning_params_default = dict(
        n_estimators=140,
        max_samples=0.45,
        contamination=0.02,
        max_features=0.69,
        bootstrap=False,
    )

    _explainer = shap.TreeExplainer

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            n_estimators=trial.suggest_int(f"{cls._name}__n_estimators", 1, 200),
            max_samples=trial.suggest_float(f"{cls._name}__max_samples", 0.0, 1.0),
            max_features=trial.suggest_float(f"{cls._name}__max_features", 0.0, 1.0),
            bootstrap=trial.suggest_categorical(
                f"{cls._name}__bootstrap", [True, False]
            ),
        )

        return cls.compile_parameters(suggestions)


@dataclass
class TabNetWrapper(TabNetClassifier):
    weights: int = 0
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    virtual_batch_size: int = 128
    drop_last: bool = True
    eval_metric: Optional[str] = None

    def fit(self, X, y):
        return super().fit(
            X_train=X.to_numpy(),
            y_train=y.to_numpy(),
            eval_metric=self.eval_metric,
            weights=self.weights,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            drop_last=self.drop_last,
        )

    def predict(self, X):
        return super().predict(X.to_numpy())

    def predict_proba(self, X):
        return super().predict_proba(X.to_numpy())

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]


class Estimator_LinearSVM(Estimator):
    _name = "LinearSVM"
    _estimator = SGDClassifier

    _requirements = dict(
        onehot=True,
        ordinal=False,
        imputation=True,
        fillna=True,
        oneclass=False,
        calibration=True,
        explanation=False,
        scaling=True,
    )

    _static_params = dict(
        max_iter=2000,
        loss="hinge",
        random_state=42,
        penalty="l2",
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate="optimal",
    )

    _tuning_params_default = dict(class_weight="balanced", alpha=0.0001,)

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(
            alpha=trial.suggest_float(f"{cls._name}__alpha", 1e-6, 1e-2, log=True),
            class_weight=trial.suggest_categorical(
                f"{cls._name}__class_weight", [None, "balanced"]
            ),
        )

        return cls.compile_parameters(suggestions)


class Estimator_OneClassSVM(Estimator):
    _name = "OneClassSVM"
    _estimator = OneClassSVMWrapper

    _requirements = dict(
        onehot=True,
        ordinal=False,
        imputation=True,
        fillna=True,
        oneclass=True,
        calibration=False,
        explanation=False,
        scaling=True,
    )

    _static_params = dict(max_iter=20000, random_state=42, learning_rate="optimal")

    _tuning_params_default = dict(nu=0.5,)

    @classmethod
    def suggest_parameters(cls, trial):
        suggestions = dict(nu=trial.suggest_float(f"{cls._name}__nu", 0.1, 1.0),)

        return cls.compile_parameters(suggestions)


# class Estimator_TabNet(Estimator):
#     _estimator = TabNetWrapper
#     _name = "TabNet"
#     _requirements = dict(
#         onehot=False,
#         ordinal=True,
#         imputation=True,
#         fillna=True,
#         oneclass=False,
#         calibration=True,
#         explanation=False,
#         scaling=False,
#     )

#     _static_params = dict(
#         optimizer_fn=torch.optim.Adam,
#         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
#         verbose=0,
#         device_name="cuda" if torch.cuda.is_available() else "cpu",
#         scheduler_params=dict(mode="min", min_lr=1e-5, factor=0.5),
#         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
#         cat_emb_dim=1,
#         max_epochs=50,
#         eval_metric="average_precision",
#         weights=1,
#         drop_last=False,
#     )

#     _tuning_params_default = dict(
#         n_d=8,
#         n_a=8,
#         n_steps=3,
#         gamma=1.2,
#         lambda_sparse=8e-4,
#         mask_type="sparsemax",
#         n_shared=3,
#         scheduler_params=dict(patience=5),
#     )

#     def __init__(self, train):
#         self._categorical_idxs, self._categorical_dims = train.describe_categories(
#             dimensions=True
#         )

#     def factory(self,):
#         return self._estimator(
#             cat_idxs=self._categorical_idxs,
#             cat_dims=[
#                 _ + 1 for _ in self._categorical_dims
#             ],  # Because we may add 1 category when we fill_na
#             **self._static_params,
#         )

#     @classmethod
#     def suggest_parameters(cls, trial):
#         suggestions = dict(
#             n_steps=trial.suggest_int(f"{cls._name}__n_steps", 1, 10),
#             n_shared=trial.suggest_int(f"{cls._name}__n_shared", 1, 10),
#             gamma=trial.suggest_float(f"{cls._name}__gamma", 1, 1.5),
#             lambda_sparse=trial.suggest_float(
#                 f"{cls._name}__lambda_sparse", 1e-6, 1e-3, log=True
#             ),
#             mask_type=trial.suggest_categorical(
#                 f"{cls._name}__mask_type", ["entmax", "sparsemax"]
#             ),
#             scheduler_params=dict(
#                 patience=trial.suggest_int(f"{cls._name}__scheduler__patience", 3, 10)
#             ),
#         )

#         n_da = trial.suggest_int(f"{cls._name}__n_da", 4, 32,)
#         suggestions["n_d"], suggestions["n_a"] = n_da, n_da

#         return cls.compile_parameters(suggestions)

#     @classmethod
#     def compile_parameters(cls, params):
#         r = {
#             **cls._static_params,
#             **cls._tuning_params_default,
#             **params,
#             "scheduler_params": {
#                 **cls._static_params["scheduler_params"],
#                 **params["scheduler_params"],
#             },
#         }
#         return {f"{cls._name}__{key}": value for key, value in r.items()}

class Resampler(Estimator):
    @classmethod
    def compile_parameters(cls, params):
        return {
            f"{cls._name}__kw_args": {
                **cls._static_params,
                **cls._tuning_params_default,
                **params,
            }
        }

    @classmethod
    def factory(cls):
        return FunctionSampler(
            func=partial(SCIData.resample, cls._estimator),
            validate=False,
            kw_args=cls._static_params,
        )

# class Resampler_SMOTE(Resampler):
#     _name = "SMOTE"
#     _estimator = SMOTENC

#     _static_params = dict(random_state=42, n_jobs=None,)

#     _tuning_params_default = dict(sampling_strategy=0.1, k_neighbors=5)

#     @classmethod
#     def suggest_parameters(cls, trial):
#         suggestions = dict(
#             sampling_strategy=trial.suggest_float(
#                 f"{cls._name}__sampling_strategy", 0.1, 0.5
#             ),
#             k_neighbors=trial.suggest_int(f"{cls._name}__k_neighbors", 2, 10),
#         )

#         return cls.compile_parameters(suggestions)

#     @classmethod
#     def factory(cls):
#         return FunctionSampler(
#             func=SCIData.SMOTE, validate=False, kw_args=cls._static_params
#         )


# class Resampler_RandomUnderSampler(Resampler):
#     _name = "RandomUnderSampler"
#     _estimator = RandomUnderSampler

#     _static_params = dict(random_state=42, replacement=False)

#     _tuning_params_default = dict(sampling_strategy=0.1)

#     @classmethod
#     def suggest_parameters(cls, trial):
#         suggestions = dict(
#             sampling_strategy=trial.suggest_float(
#                 f"{cls._name}__sampling_strategy", 0.05, 0.5
#             )
#         )

#         return cls.compile_parameters(suggestions)


# class No_Resampling(Resampler):
#     _name = "No_Resampling"

#     @classmethod
#     def suggest_parameters(cls, trial):
#         return dict()

#     @staticmethod
#     def _(X, y):
#         return X, y

#     @classmethod
#     def factory(cls):
#         return FunctionSampler(func=cls._, validate=False)
