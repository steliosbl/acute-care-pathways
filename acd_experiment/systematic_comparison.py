import warnings, pickle, os, itertools, argparse
from joblib import Parallel, delayed, parallel_backend

import numpy as np
import pandas as pd

import optuna, sqlalchemy

optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV

from .models import *
from .utils.systematic_comparison import get_metrics, get_threshold_fpr
from .base_dataset import BaseDataset

from typing import Dict, Any, Iterable, Optional, Tuple, Callable


def oneclass_split(X_train, y_train, kf):
    for train_index, test_index in kf.split(X_train, y_train):
        if type(y_train) == pd.Series:
            y_train = y_train.to_numpy()
        trainy, validy = (
            y_train[train_index],
            y_train[test_index],
        )

        test_index = np.concatenate((train_index[trainy], test_index))
        train_index = train_index[~trainy]
        yield train_index, test_index


def study_grid_from_args(args, scii):
    estimators = dict(
        all=[
            #      Estimator_IsolationForest,
            Estimator_LightGBM,
            #      Estimator_LogisticRegression,
            #      Estimator_RandomForest,
            Estimator_XGBoost,
            Estimator_LinearSVM,
            #      Estimator_OneClassSVM,
            Estimator_L1Regression,
            Estimator_L2Regression,
            Estimator_ElasticNetRegression,
        ]
    )
    estimators = estimators | {_._name: [_] for _ in estimators["all"]}

    features = scii.feature_group_combinations(scored=args['scored_features'])

    r = dict(estimators=estimators[args["models"]], features=features, scii=scii)

    if args["time_thresholds"]:
        r = dict(resamplers=[None], outcome_thresholds=list(range(1, 31))) | r
    else:
        r = dict(resamplers=[None], outcome_thresholds=[1]) | r

    return study_grid(**r)


def study_grid(estimators, resamplers, features, scii, outcome_thresholds):
    oneclass_estimators, binary_estimators = (
        [_ for _ in estimators if _._requirements["oneclass"]],
        [_ for _ in estimators if not _._requirements["oneclass"]],
    )
    if len(features) == 0:
        features = scii.feature_group_combinations
    categorical_feature_groups = [
        k for k, v in features.items() if (scii[v].dtypes == "category").all()
    ]

    r = []
    for (estimator, resampler, feature_name, outcome_threshold) in itertools.product(
        estimators, resamplers, features, outcome_thresholds
    ):
        if estimator._requirements["oneclass"] and resampler is not None:
            continue
        if feature_name in categorical_feature_groups and resampler is not None:
            continue
        r.append(
            dict(
                estimator=estimator,
                resampler=resampler,
                feature_group=feature_name,
                features=features[feature_name],
                outcome_within=outcome_threshold,
            )
        )

    return r


@dataclass
class PipelineFactory:
    estimator: Estimator
    resampler: Optional[Resampler]
    X_train: BaseDataset
    y_train: pd.Series

    def __post_init__(self):
        self._scaler = self.estimator._requirements["scaling"]

    def __call__(self, **kwargs):
        steps = []
        if self.resampler is not None:
            steps.append((self.resampler._name, self.resampler.factory(),))

        if self._scaler:
            steps.append(
                (
                    "Scaling",
                    ColumnTransformer(
                        [("Scaler", StandardScaler(), self.X_train.numeric_columns)],
                        remainder="passthrough",
                    ),
                )
            )

        steps.append((self.estimator._name, self.estimator.factory()))
        return ImbPipeline(steps=steps).set_params(**kwargs)


@dataclass
class Objective:
    estimator: Estimator
    resampler: Optional[Resampler]
    pipeline_factory: PipelineFactory
    X_train: BaseDataset
    y_train: pd.Series
    cv: int
    cv_jobs: int
    scoring: str
    n_trials: int
    stop_callback: Callable

    def __post_init__(self):
        self._fit_params = self.estimator.fit_params(self.X_train, self.y_train)

    def __call__(self, trial):
        trial_params = {
            **(self.resampler.suggest_parameters(trial) if self.resampler else {}),
            **self.estimator.suggest_parameters(trial),
        }
        model = self.pipeline_factory(**trial_params)

        cv = self.cv
        if self.estimator._requirements["oneclass"]:
            cv = oneclass_split(
                self.X_train,
                self.y_train,
                StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
            )

        score = cross_validate(
            model,
            self.X_train,
            self.y_train,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.cv_jobs,
            fit_params=self._fit_params,
        )["test_score"].mean()

        if trial.number >= self.n_trials:
            self.stop_callback()

        return score


def restructure_parameters(params, estimator_name, resampler_name):
    return {k: v for k, v in params.items() if k.startswith(estimator_name)} | {
        f"{resampler_name}__kw_args": {
            k.split("__")[1]: v
            for k, v in params.items()
            if k.startswith(resampler_name)
        }
    }


def evaluate_model(model, X_test, y_test, n_resamples):
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_pred_proba = model.decision_function(X_test)

    y_pred = np.where(
        y_pred_proba > get_threshold_fpr(y_test, y_pred_proba, target=0.05), 1, 0
    )

    metrics = get_metrics(y_test, y_pred, y_pred_proba, n_resamples)

    return metrics, y_pred_proba


def get_xy(scii, estimator, features, outcome_within=1, outcome="CriticalEvent"):
    sci_args = dict(
        x=features,
        imputation=estimator._requirements["imputation"],
        onehot_encoding=estimator._requirements["onehot"],
        ordinal_encoding=estimator._requirements["ordinal"],
        fillna=estimator._requirements["fillna"],
        outcome_within=outcome_within,
        outcome=outcome,
    )

    X, y = scii.xy(**sci_args)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, shuffle=False
    )
    return type(scii)(X_train), type(scii)(X_test), y_train, y_test


def construct_study(
    estimator: Estimator,
    resampler: Optional[Resampler],
    scii: BaseDataset,
    feature_group: str,
    features: Iterable[str],
    outcome_within: int,
    cv=5,
    scoring="average_precision",
    storage=None,
    model_persistence_path=None,
    cv_jobs=1,
    n_trials=100,
    outcome="CriticalEvent",
    **kwargs,
):
    X_train, X_test, y_train, y_test = get_xy(
        scii, estimator, features, outcome_within, outcome
    )
    name = f"{estimator._name}_{resampler._name if resampler else 'None'}_Within-{outcome_within}_{feature_group}"
    study = optuna.create_study(
        direction="maximize", study_name=name, storage=storage, load_if_exists=True
    )

    pipeline_factory = PipelineFactory(
        estimator=estimator, resampler=resampler, X_train=X_train, y_train=y_train,
    )

    objective = Objective(
        estimator=estimator,
        resampler=resampler,
        pipeline_factory=pipeline_factory,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        cv_jobs=cv_jobs,
        n_trials=n_trials,
        stop_callback=study.stop,
    )

    def handle_study_result(model_persistence_path=None, n_resamples=99, **kwargs):
        params = study.best_params
        if resampler:
            params = restructure_parameters(params, estimator._name, resampler._name)

        X, y = X_train, y_train
        if estimator._requirements["oneclass"]:
            X = type(scii)(X_train[y_train.eq(0)])
            y = y_train[y_train.eq(0)]

        explanations = []
        explain = estimator._requirements["explanation"] and (
            model_persistence_path is not None
        )
        if estimator._requirements["calibration"]:
            model = CalibratedClassifierCV(
                pipeline_factory(**params), cv=cv, method="isotonic", n_jobs=cv_jobs,
            ).fit(X, y)
            if explain:
                explanations = estimator.explain_calibrated(
                    model, X, X_test, cv_jobs=cv_jobs
                )
        else:
            model = pipeline_factory(**params).fit(X, y)
            if explain:
                explanations = estimator.explain(model[estimator._name], X, X_test)

        if model_persistence_path is not None:
            with open(f"{model_persistence_path}/{name}.bin", "wb") as file:
                pickle.dump((model, explanations), file)

        metrics, y_pred_proba = evaluate_model(model, X_test, y_test, n_resamples)

        metrics = (
            dict(
                name=name,
                estimator=estimator._name,
                resampler=resampler._name if resampler else "None",
                features=feature_group,
                outcome_within=outcome_within,
            )
            | metrics
        )

        return metrics, pd.Series(y_pred_proba, index=y_test.index, name=name)

    def call(model_persistence_path=None, n_resamples=99, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            while True:
                try:
                    study.optimize(objective, **kwargs)
                    return handle_study_result(model_persistence_path, n_resamples)
                except (
                    optuna.exceptions.StorageInternalError,
                    sqlalchemy.exc.OperationalError,
                    AssertionError,
                ):
                    print("################# CAUGHT DB ERROR #################")
                    pass

    return call


def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--models", help="Can be 'all', 'LightGBM', etc.", type=str, default="all"
    )
    parser.add_argument(
        "-j", "--njobs", help="Number of CPUs to use. Default=1", type=int, default=1
    )
    parser.add_argument(
        "-c", "--cv_jobs", help="Number of CV jobs. Default=5", type=int, default=5
    )
    parser.add_argument(
        "-p",
        "--persist",
        help="Filepath to save the models. If unset, wont save them",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to only use a small subset of data for debugging",
        action="store_true",
    )
    parser.add_argument(
        "--time_thresholds",
        help="Whether to run the time threshold test",
        action="store_true",
    )
    parser.add_argument(
        "-t", "--trials", help="Number of trials. Default=1000", type=int, default=100
    )
    parser.add_argument(
        "-hr", "--hours", help="Trial timeout in hours", type=int, default=2
    )
    parser.add_argument(
        "-s",
        "--storage",
        help="Trial storage for optuna",
        default=None,  # "sqlite:///models/studies.db",
    )
    parser.add_argument(
        "-o", "--output", help="Output path for final results", default="results.h5"
    )
    parser.add_argument(
        "--n_resamples",
        help="Number of resamples for bootstrapping metrics",
        default=999,
    )
    parser.add_argument("-v", "--verbose", help="Optuna verbosity", action="store_true")
    parser.add_argument(
        "--scored_features", help="Limit feature groups", action="store_true", default=False
    )
    parser.add_argument("--outcome", help="The target label", default="CriticalEvent")

    return parser


def run_experiment(args, scii):
    args = vars(args)

    if args["verbose"]:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    if args["persist"] is not None:
        try:
            os.makedirs(args["persist"])
        except FileExistsError:
            pass

    if args["debug"]:
        scii = type(scii)(scii.sample(10000))
        args['trials'] = 2
        args['n_resamples'] = 10
        args['vecbose'] = True

    if args["storage"] is not None:
        args["storage"] = optuna.storages.RDBStorage(
            url=args["storage"], engine_kwargs={"connect_args": {"timeout": 100}}
        )

    studies = [
        construct_study(**_, **args, scii=scii, n_trials=args["trials"])
        for _ in study_grid_from_args(args, scii.omit_redundant().categorize())
    ]

    study_args = dict(
        model_persistence_path=args["persist"],
        n_resamples=args["n_resamples"],
        n_trials=args["trials"],
        timeout=args["hours"] * 60 * 60,
    )

    if args["njobs"] > 1:
        print("Starting execution (parallel)")
        with parallel_backend("loky", inner_max_num_threads=args["cv_jobs"]):
            results = Parallel(n_jobs=args["njobs"])(
                delayed(_)(**study_args) for _ in studies
            )
    else:
        print("Starting execution (linear)")
        results = [_(**study_args) for _ in studies]

    metrics, y_preds = list(zip(*results))
    pd.DataFrame(metrics).to_hdf(args["output"], "metrics")
    pd.concat(y_preds, axis=1).to_hdf(args["output"], "y_preds")
