# %%
import os, pickle, warnings, itertools
from pathlib import Path
from functools import partial 

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
warnings.simplefilter(action='ignore', category=FutureWarning)


from salford_datasets.salford import SalfordData, SalfordFeatures, SalfordPrettyPrint, SalfordCombinations
from acd_experiment.salford_adapter import SalfordAdapter

# %%
class Notebook:
    DATA_DIR = Path('data/Salford')
    CACHE_DIR = Path('data/cache')
    IMAGE_DIR = Path('images/shallow')
    SYSTEMATIC_COMPARISON_DIR = Path('data/systematic_comparison/')
    RE_DERIVE = False

# %%
from transformer_experiment.utils.shallow_classifiers import load_salford_dataset, get_train_test_indexes

SAL = load_salford_dataset(Notebook.RE_DERIVE, Notebook.DATA_DIR)
SAL_TRAIN_IDX, SAL_TEST_IDX, SAL_TEST_UNSEEN_IDX, SAL_TEST_IS_UNSEEN = get_train_test_indexes(SAL)

# %%
from acd_experiment.models import Estimator_L1Regression, Estimator_LinearSVM, Estimator_LightGBM, Estimator_L2Regression, Estimator_ElasticNetRegression, Estimator_XGBoost

FEATURE_GROUP_CORRESPONDENCE = {
    'news': 'news',
    'with_phenotype': 'news_with_phenotype',
    'with_labs': 'with_labs',
    'with_services': 'with_notes_labs_and_hospital'
}

PRETTY_PRINT_FEATURE_GROUPS = {
    'Reference': 'Reference',
    'news': 'Vitals',
    'with_phenotype': '& Obs',
    'with_labs': '& Labs',
    'with_services': '& Services',
}

PRETTY_PRINT_ESTIMATORS = dict(
    NEWS2='NEWS2',
    LogisticRegression='LR',
    L1Regression='LR-L1',
    L2Regression='LR-L2',
    ElasticNetRegression='LR-EN',
    XGBoost='XGBoost',
    LightGBM='LightGBM',
    LinearSVM='LinearSVM'
)

PRETTY_PRINT_FEATURE_NAMES = {
    k:(
        v.replace('First Blood ', '')
        .replace('First Obs ', '')
        .replace('Emergency Department', 'ED')
    )
    for k,v in SalfordPrettyPrint.items()
}

ESTIMATORS = {_._name: _ for _ in [
    Estimator_LightGBM,
    Estimator_L2Regression,
    Estimator_XGBoost,
    Estimator_LinearSVM,
    Estimator_L1Regression,
    Estimator_ElasticNetRegression,
]}

STUDY_GRID = list(itertools.product(ESTIMATORS.keys(), FEATURE_GROUP_CORRESPONDENCE.keys()))

# %% [markdown]
# ## Model Training

# %%
from acd_experiment.salford_adapter import SalfordAdapter
from sklearn.calibration import CalibratedClassifierCV
import optuna
from acd_experiment.systematic_comparison import get_xy, PipelineFactory

def run_pretuned(sal, estimator_name, feature_group_name, cv_jobs=4, explain_models=['LightGBM', 'L2Regression'], feature_columns=None, outcome_within=1):
    params = optuna.load_study(
        study_name =f'{estimator_name}_None_Within-1_{FEATURE_GROUP_CORRESPONDENCE[feature_group_name]}', storage=f'sqlite:///{Notebook.SYSTEMATIC_COMPARISON_DIR}/{estimator_name}.db'
    ).best_params
    
    estimator = ESTIMATORS[estimator_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = SalfordAdapter(sal).xy(
            x=feature_columns or SalfordCombinations[feature_group_name],
            imputation = estimator._requirements['imputation'],
            fillna = estimator._requirements['fillna'],
            ordinal_encoding = estimator._requirements['ordinal'],
            onehot_encoding = estimator._requirements['onehot'],
            outcome_within=outcome_within
        )
    X_train, y_train = SalfordAdapter(X.loc[SAL_TRAIN_IDX]), y.loc[SAL_TRAIN_IDX].values

    pipeline_factory = PipelineFactory(
        estimator=estimator, resampler=None, X_train=X_train, y_train=y_train,
    )

    model = CalibratedClassifierCV(
        pipeline_factory(**params), cv=cv_jobs, method="isotonic", n_jobs=cv_jobs,
    ).fit(X_train, y_train)


    y_pred_proba = model.predict_proba(X.loc[SAL_TEST_IDX])[:,1]
    y_pred_proba_unseen = y_pred_proba[SAL_TEST_IS_UNSEEN]

    explanations = None
    if estimator_name in explain_models:
        explanations = estimator.explain_calibrated(
            model, X_train, SalfordAdapter(X.loc[SAL_TEST_IDX]), cv_jobs=cv_jobs
        )

    return y_pred_proba, y_pred_proba_unseen, explanations, model
    
if Notebook.RE_DERIVE:
    RESULTS = {}
    for estimator_name, feature_group_name in (pbar := tqdm(STUDY_GRID)):
        pbar.set_description(f'Training {estimator_name} on {feature_group_name}')
        RESULTS[(PRETTY_PRINT_ESTIMATORS[estimator_name], PRETTY_PRINT_FEATURE_GROUPS[feature_group_name])] = run_pretuned(SAL, estimator_name, feature_group_name)

        with open(Notebook.CACHE_DIR/'shallow_results_2.bin', 'wb') as file:
            pickle.dump(RESULTS, file)
else:
    with open(Notebook.CACHE_DIR/'shallow_results_2.bin', 'rb') as file:
            RESULTS = pickle.load(file)

# %%
from transformer_experiment.utils.shallow_classifiers import get_discriminative_metrics
Y_TRUES = {
    'Complete': SAL.CriticalEvent.loc[SAL_TEST_IDX],
    'Unseen': SAL.CriticalEvent.loc[SAL_TEST_UNSEEN_IDX],
}
def get_full_metrics_tables(results):
    metrics = {
        'Complete': [],
        'Unseen': [],
    }
    for (estimator_name, feature_group_name), y_preds in results.items():
        for y_pred_proba, (y_true_name, y_true) in zip(y_preds, Y_TRUES.items()):
            metrics[y_true_name].append(dict(
                Estimator = estimator_name,
                Features = feature_group_name,
            ) | get_discriminative_metrics(
                y_true, y_pred_proba
            ))
    
    for y_true_name, y_true in Y_TRUES.items():
        metrics[y_true_name].append(dict(
            Estimator='NEWS2',
            Features='Reference'
        ) | get_discriminative_metrics(
            y_true, SAL.NEWS_Score_Admission.loc[y_true.index]
        ))

    return {
        y_true_name: pd.DataFrame(metric_list) for y_true_name, metric_list in metrics.items()
    }

if Notebook.RE_DERIVE:
    METRICS = get_full_metrics_tables(RESULTS)
    with open(Notebook.CACHE_DIR/'shallow_results_2_metrics.bin', 'wb') as file:
        pickle.dump(METRICS, file)
else:
    with open(Notebook.CACHE_DIR/'shallow_results_2_metrics.bin', 'rb') as file:
            METRICS = pickle.load(file)

# %% [markdown]
# ## Transformer Results

# %%
def load_transformer_results():
    with open(Notebook.CACHE_DIR/'transformer_shallow_results_ensemble.bin', 'rb') as file:
        shallow_results_ensemble = pickle.load(file)
    with open(Notebook.CACHE_DIR/'transformer_shallow_results_embonly.bin', 'rb') as file:
        shallow_results_joint = pickle.load(file)
    with open(Notebook.CACHE_DIR/'transformer_finetuning_results.bin', 'rb') as file:
        finetuning_results = pickle.load(file)

    pretty_print_ensembles = PRETTY_PRINT_ESTIMATORS | {'LightGBM': 'LGBM'}
    shallow_results = {
        (PRETTY_PRINT_ESTIMATORS[key[0]], key[1]): value for key, value in shallow_results_joint.items()
    }| {
        (f'{pretty_print_ensembles[key[0]]}-Ensemble', key[2]): value for key, value in shallow_results_ensemble.items()
        if key[0] == key[1]
    }
    
    return shallow_results, finetuning_results

TRANSF_RESULTS_SHALLOW, TRANSF_RESULTS_DEEP = load_transformer_results()
TRANSF_METRICS_SHALLOW, TRANSF_METRICS_DEEP = get_full_metrics_tables(TRANSF_RESULTS_SHALLOW), get_full_metrics_tables(TRANSF_RESULTS_DEEP)

# %%
def get_select_transformer_results(results_shallow, results_deep):
    shallow_choices = [('LightGBM', 'BioClinicalBert'), ('LR-L2', 'BioClinicalBert'), ('LGBM-Ensemble', 'BioClinicalBert')]
    deep_choices = [('BioClinicalBert', 'All'), ('PubMedBert', 'All'), ('Bert', 'All')]
    shallow_choices = {_:results_shallow[_] for _ in shallow_choices}
    deep_choices = {_: results_deep[_] for _ in deep_choices}
    return shallow_choices | deep_choices

TRANSF_RESULTS_SELECT = get_select_transformer_results(TRANSF_RESULTS_SHALLOW, TRANSF_RESULTS_DEEP)
TRANSF_METRICS_SELECT = get_full_metrics_tables(TRANSF_RESULTS_SELECT)

# %%
TRANSF_METRICS_SELECT['Unseen']

# %% [markdown]
# ## Metrics

# %% [markdown]
# ### Summary Table

# %%
def summary_metrics_select_estimators(metrics, estimators=['LightGBM', 'LR-L2']):
    df = pd.DataFrame(dict(
            Metric=metric,
            Estimator=_['Estimator'],
            Features=_['Features'],
            Value=f"{_[metric]:.3f}",
            Dataset=dataset
        ) for dataset, df in metrics.items() for _ in df[df.Estimator.isin(estimators)].to_dict(orient='records') for metric in ('AUROC', 'AP')
    ).sort_values(['Metric', 'Estimator', 'Dataset']).set_index(['Metric', 'Estimator', 'Dataset', 'Features']).unstack()['Value'][list(PRETTY_PRINT_FEATURE_GROUPS.values())[1:]]
    return df

# %%
#print(result_metrics_summary.to_latex(bold_rows=True, escape=False, column_format='lll|cccc', multirow=True, formatters=[lambda x: f'${x}$' for _ in range(result_metrics_summary.shape[1])]))

# %%
def summary_metrics_transformers(metrics_shallow, metrics_deep, estimators=['LR-L2', 'LGBM-Ensemble'], transformers=['BioClinicalBert', 'PubMedBert', 'Bert']):
    shallow_rows = [dict(
            Metric=metric,
            Estimator=_['Estimator'],
            Transformer=_['Features'],
            Value=f"{_[metric]:.3f}",
            Dataset=dataset
        ) 
        for dataset, df in metrics_shallow.items() 
        for _ in df[df.Estimator.isin(estimators) & df.Features.isin(transformers)].to_dict(orient='records') 
        for metric in ('AUROC', 'AP')
    ]

    deep_rows = [dict(
            Metric=metric,
            Estimator='WFine-Tuning',
            Transformer=_['Estimator'],
            Value=f"{_[metric]:.3f}",
            Dataset=dataset
        ) for dataset, df in metrics_deep.items() for _ in df[df.Estimator.isin(transformers) & (df.Features=='All')].to_dict(orient='records') for metric in ('AUROC', 'AP')
    ]
    
    df = pd.DataFrame(shallow_rows + deep_rows).sort_values(['Metric', 'Estimator', 'Dataset']).set_index(['Metric', 'Estimator', 'Dataset', 'Transformer']).unstack()['Value']
    return df

# %%
#print(summary_metrics_transformers.to_latex(bold_rows=True, escape=False, column_format='lll|ccc', multirow=True, formatters=[lambda x: f'${x}$' for _ in range(summary_metrics_transformers.shape[1])]))

# %% [markdown]
# ### Detailed Tables

# %%
def detailed_metrics_all_estimators(metrics, index=['Features', 'Estimator']):
    df = pd.DataFrame(dict(
            Dataset=dataset,
            Metric=metric,
            Estimator=_['Estimator'],
            Features=_['Features'],
            Summary = f'{_[metric]:.4f} ({_[metric+"_Lower"]:.4f}-{_[metric + "_Upper"]:.4f})'
        ) for dataset, df in metrics.items() for _ in df.to_dict(orient='records') for metric in ('AUROC', 'AP')
    ).pivot(index=index, columns=['Metric', 'Dataset'], values='Summary')

    return df[['AUROC', 'AP']]

# %%
def metrics_delta_comparison(metrics):
    print(f"AUROC: {(metrics['Complete']['AUROC'] - metrics['Unseen']['AUROC']).median()}")
    print(f"AP: {(metrics['Complete']['AP'] - metrics['Unseen']['AP']).median()}")

metrics_delta_comparison(METRICS)

# %%
#print(result_metrics_long.to_latex(bold_rows=True, multirow=True, multicolumn=True, longtable=False, column_format='llrrrr'))

# %%
# %%
#print(result_metrics_transf_deep_long.to_latex(bold_rows=True, multirow=True, multicolumn=True, longtable=False, column_format='llrrrr'))

# %%
metrics_delta_comparison(TRANSF_METRICS_SHALLOW)

# %%

# %%
# %% [markdown]
# ### NEWS Comparison

# %%
from transformer_experiment.utils.shallow_classifiers import get_decision_metrics, get_decision_threshold
from sklearn.metrics import recall_score

def news_threshold_comparison(y_preds, estimators=['LightGBM', 'L2Regression'], feature_groups=PRETTY_PRINT_FEATURE_GROUPS, news_thresholds=[3, 5, 7], confidence_intervals=True, summary=False):
    y_true = Y_TRUES['Complete']
    news_values = SAL.loc[y_true.index].NEWS_Score_Admission
    news_sensitivities = {
        threshold: recall_score(y_true, news_values >= threshold) 
        for threshold in news_thresholds
    }

    results = [
        dict(
            Estimator="NEWS",
            Features=None,
            Threshold=f'$\geq {news_threshold}$',
        ) | get_decision_metrics(y_true, news_values >= news_threshold, confidence_intervals) 
        for news_threshold in news_sensitivities.keys()
    ]

    for estimator in estimators:
        for feature_group, feature_group_name in feature_groups.items():
            if feature_group == 'Reference':
                continue
            y_pred_proba = y_preds[(estimator, feature_group)][0]
            for news_threshold, observed_sensitivity in news_sensitivities.items():
                threshold = get_decision_threshold(y_true, y_pred_proba, target_recall=observed_sensitivity)
                y_pred = np.where(y_pred_proba > threshold, 1, 0)
                results.append(dict(
                    Estimator=estimator,
                    Features=feature_group_name,
                    Threshold=f'$\geq {threshold:.3f}$',
                ) | get_decision_metrics(y_true, y_pred, confidence_intervals))

    result = pd.DataFrame(results).set_index(['Features', 'Estimator', 'Threshold'])#[list(r.columns[-1:]) + list(r.columns[:-1])]
    if not summary:
        return result.loc[feature_groups.values()]
    
    news_mask = result.index.get_level_values('Estimator') == 'NEWS'
    model_mask = result.index.get_level_values('Estimator').isin(['LightGBM', 'L2Regression'])
    feature_mask = result.index.get_level_values('Features') == '& Services'
    result = result[news_mask | (model_mask & feature_mask)]
    result.index = result.index.droplevel(0)
    return result

# %%
from transformer_experiment.utils.shallow_classifiers import get_decision_metrics, get_decision_threshold
from sklearn.metrics import recall_score

def news_threshold_comparison_transformers(y_preds, estimators=['LGBM-Ensemble', 'PubMedBert'], news_thresholds=[3, 5, 7], confidence_intervals=True, summary=False):
    y_true = Y_TRUES['Complete']
    news_values = SAL.loc[y_true.index].NEWS_Score_Admission
    news_sensitivities = {
        threshold: recall_score(y_true, news_values >= threshold) 
        for threshold in news_thresholds
    }

    results = [
        dict(
            Estimator="NEWS",
            Features=None,
            Threshold=f'$\geq {news_threshold}$',
        ) | get_decision_metrics(y_true, news_values >= news_threshold, confidence_intervals) 
        for news_threshold in news_sensitivities.keys()
    ]

    for estimator, feature_group in y_preds.keys():
        if not estimator in estimators:
            continue
    
        y_pred_proba = y_preds[(estimator, feature_group)][0]
        for news_threshold, observed_sensitivity in news_sensitivities.items():
            threshold = get_decision_threshold(y_true, y_pred_proba, target_recall=observed_sensitivity)
            y_pred = np.where(y_pred_proba > threshold, 1, 0)
            results.append(dict(
                Estimator=estimator,
                Features=feature_group,
                Threshold=f'$\geq {threshold:.3f}$',
            ) | get_decision_metrics(y_true, y_pred, confidence_intervals))

    result = pd.DataFrame(results).set_index(['Estimator', 'Features', 'Threshold'])#[list(r.columns[-1:]) + list(r.columns[:-1])]
    if not summary:
        return result.loc[['NEWS'] + estimators]
    
    news_mask = result.index.get_level_values('Estimator') == 'NEWS'
    feature_mask = result.index.get_level_values('Features').isin(['PubMedBert', 'All'])
    result = result[news_mask | feature_mask]
    result.index = result.index.droplevel(1)
    return result

# %%
#print(result_news_comparison_transf.to_latex(multirow=True, escape=False, formatters=[lambda _: f'${_:.4f}$'] * result_news_comparison_transf.shape[1]))

# %% [markdown]
# ## Bar Plots

# %%
def select_feature_ablation_barplot(metrics_df, metric='AUROC', ylim=(0.0, 1.0), ylabel=None, title=None, save=None):
    sns.set_style('whitegrid')
    df = metrics_df[metrics_df.Estimator != 'NEWS2'].copy()
    err = (df[f'{metric}_Upper']-df[f'{metric}_Lower'])/2
    df['Features'] = pd.Categorical(
        df.Features.replace(PRETTY_PRINT_FEATURE_GROUPS), 
        ordered=True, categories=list(PRETTY_PRINT_FEATURE_GROUPS.values())[1:]
    )

    g = sns.catplot(
        data=df.sort_values(['Estimator','Features']), x='Features', y=metric, hue='Estimator',
        kind='bar', palette='muted', height=6, orient='v', legend=False
    )
    g.despine(left=True)
    g.fig.set_size_inches(9,3)
    g.ax.set_ylim(ylim)

    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.ax.patches]
    y_coords = [p.get_height() for p in g.ax.patches]
    g.ax.errorbar(x=x_coords, y=y_coords, yerr=err, fmt="none", c= "k")

    news_performance = metrics_df.loc[metrics_df.Estimator == 'NEWS2', metric].values[0]

    g.refline(y = news_performance , color = 'gray', linestyle = '--', label = "Reference (NEWS2)") 
    g.add_legend(title='Estimator')
    
    g.set_xlabels("Feature Set", labelpad=20)
    g.ax.set_title(title, fontsize=16)
    
    if ylabel:
        g.set_ylabels(ylabel)

    if save:
        plt.savefig(save, bbox_inches="tight", format='svg')

# %% [markdown]
# ## Calibration Curves

# %%
def estimator_y_preds_across_feature_groups(y_preds, estimator_target='LightGBM'):
    return {
        PRETTY_PRINT_FEATURE_GROUPS[feature_group]: y_pred_proba 
        for (estimator, feature_group), (y_pred_proba, *_) in y_preds.items()
        if estimator == estimator_target
    }

def feature_group_y_preds_across_estimators(y_preds, feature_group_target='with_services'):
    return {
        estimator: y_pred_proba
        for (estimator, feature_group), (y_pred_proba, *_) in y_preds.items()
        if feature_group == feature_group_target
    }

def all_y_preds_by_estimator(y_preds):
    return {
        estimator: y_pred_proba
        for (estimator, feature_group), (y_pred_proba, *_) in y_preds.items()
    }


# %%
import matplotlib.gridspec
from transformer_experiment.utils.plots import plot_calibration_curves

def calib_curves(y_preds, y_preds_transf, save=None):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10,10))
    gs = matplotlib.gridspec.GridSpec(4, 4, hspace=0.6, wspace=0.4)
    ax1 = plt.subplot(gs[:2, :2])
    ax2 = plt.subplot(gs[:2, 2:])
    ax3 = plt.subplot(gs[2:4, 1:3])
    
    y_true = Y_TRUES['Complete']
    y_preds_l, y_preds_r, y_preds_d = (
        estimator_y_preds_across_feature_groups(y_preds, 'LightGBM'), 
        feature_group_y_preds_across_estimators(y_preds, 'with_services'),
        all_y_preds_by_estimator(y_preds_transf)
    )
    y_preds_r['LinearSVM'] = estimator_y_preds_across_feature_groups(y_preds, 'XGBoost')['& Labs']

    plot_calibration_curves(y_true, y_preds_l, ax=ax1, title='(a) GBDT (LightGBM)')
    plot_calibration_curves(y_true, y_preds_r, ax=ax2, title='(b) All Models, Complete Feature Set')
    plot_calibration_curves(y_true, y_preds_d, ax=ax3, title='(c) Language Modelling')

    sns.move_legend(ax1, "upper left", bbox_to_anchor=(2.2, 1.1), frameon=False, title='(a) GBDT')
    sns.move_legend(ax2, "lower left", bbox_to_anchor=(1, -0.1), frameon=False, title='(b) All models')
    sns.move_legend(ax3, "center left", bbox_to_anchor=(1, .5), frameon=False, title='(c) Language Modelling')
    ax2.set_ylabel(None)
   
   # fig.tight_layout(h_pad=0.5, w_pad=0.5)
    fig.subplots_adjust(top=0.9)
    fig.suptitle('Model Calibration')
    if save:
        plt.savefig(save, bbox_inches='tight', format='svg')
   
# %% [markdown]
# ## Alert Rate & PR

# %%
TEST_SET_N_DAYS = (
    SAL.loc[SAL_TEST_IDX].AdmissionDate.max() - SAL.loc[SAL_TEST_IDX].AdmissionDate.min()
).days
TEST_SET_N_DAYS


# %%
from transformer_experiment.utils.plots import plot_pr_curves, plot_alert_rate

def pr_and_alertrate_curves_estimator(y_preds, estimator, title, legend_label, subtitle_lettering, save=None):
    y_preds = estimator_y_preds_across_feature_groups(y_preds, estimator)   
    return pr_and_alertrate_curves(y_preds, title, legend_label, subtitle_lettering, save)

def pr_and_alertrate_curves_feature_group(y_preds, feature_group, title, legend_label, subtitle_lettering, save=None):
    y_preds = feature_group_y_preds_across_estimators(y_preds, feature_group)
    y_preds = {
        PRETTY_PRINT_ESTIMATORS[estimator]: y for estimator, y in y_preds.items()
    }
    return pr_and_alertrate_curves(y_preds, title, legend_label, subtitle_lettering, save)

def pr_and_alertrate_curves_transformers(y_preds_transf, title, legend_label, subtitle_lettering, save=None):
    y_preds = all_y_preds_by_estimator(y_preds_transf)
    return pr_and_alertrate_curves(y_preds, title, legend_label, subtitle_lettering, save)

def pr_and_alertrate_curves(y_preds, title, legend_label, subtitle_lettering, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.set_style("whitegrid")
    for _ in ax:
        _.set_box_aspect(1)

    y_true = Y_TRUES['Complete']
    baseline_news = {'NEWS2': SAL.loc[y_true.index].NEWS_Score_Admission}
    
    with sns.color_palette('muted'):
        plot_alert_rate(
            y_true, y_preds | baseline_news, TEST_SET_N_DAYS,
            ax=ax[0], intercepts=False, baseline_key='NEWS2', 
            title=f'{subtitle_lettering[0]} Alert Rate vs. Sensitivity', xlim=(0.4, 1.0), ylim=(0, 30)
        )

        plot_pr_curves(
            y_true, y_preds | baseline_news, 
            smoothing=True, ax=ax[1], 
            palette=sns.color_palette('muted'), 
            baseline_key='NEWS2', 
            title=f'{subtitle_lettering[1]} Precision-Recall'
        )
        
    sns.move_legend(ax[1], "center left", bbox_to_anchor=(1, 0.5), frameon=False, title=legend_label)
    ax[0].legend([], [], frameon=False)
    
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=14)

    if save:
        plt.savefig(save, bbox_inches='tight', format='svg')

# %% [markdown]
# ### Intercepts

# %%
from transformer_experiment.utils.plots import plot_alert_rate, biggest_alert_rate_diff

def plot_alert_rate_with_intercepts(y_preds):
    y_true = Y_TRUES['Complete']
    baseline_news = {'NEWS2': SAL.loc[y_true.index].NEWS_Score_Admission}

    plot_alert_rate(
        y_true, {'Model': y_preds} | baseline_news, TEST_SET_N_DAYS,
        intercepts=True, baseline_key='NEWS2', 
    )

    sensitivity, news_rate, lgbm_rate = biggest_alert_rate_diff(
        y_true, baseline_news['NEWS2'], y_preds, TEST_SET_N_DAYS,
    )

    print(
        f"At sensitivity ~{sensitivity:.3f}: NEWS Alert rate: {news_rate:.3f}, Model Alert rate: {lgbm_rate:.3f} -> {100-(lgbm_rate*100/news_rate):.3f}% less"
    )

plot_alert_rate_with_intercepts(
    estimator_y_preds_across_feature_groups(RESULTS, 'LightGBM')['& Services']
)

# %%
from transformer_experiment.utils.plots import plot_shap_features_joint

def lgbm_shap_beeswarm(y_preds, title='Feature Interactions - GBDT (LightGBM)', save=None):
    explanations = y_preds[('LightGBM', 'with_services')][-2]
    explanations.values[explanations.values >= 2.8] = 2.8
    plot_shap_features_joint(
        explanations, 
        max_display=250,
        figsize=(12, 8),
        wspace=-0.25,
        bar_aspect=0.04,
        topadjust=0.925,
        title=title,
        save=save
    )

# %% [markdown]
# ### Categorical Features

# %%
import shap
def categorical_shap_bars_lgbm(y_preds, save=None):
    explanations = y_preds[('LightGBM', 'with_services')][-2]
    
    r = []
    for column in SAL.categorical_columns(SalfordCombinations['with_services']):
        idx = explanations.feature_names.index(PRETTY_PRINT_FEATURE_NAMES[column])
        df = pd.DataFrame(
            zip(explanations[:, idx].values, explanations[:, idx].data), 
            columns=['Value', 'Data']).pivot(columns='Data').Value
        selected = df.apply(abs).mean().sort_values().tail(10).index
        r.append(df.mean().loc[selected].sort_values().rename('SHAP').to_frame().assign(Feature=PRETTY_PRINT_FEATURE_NAMES[column]))

    df = pd.concat(r).reset_index()

    #df['Data'] = df['Data'].replace(pretty_print_categoricals)
    #df['Feature'] = df['Feature'].replace({'A&E Diagnosis': 'ED Diagnosis'})
    df['Hue'] = (df.SHAP > 0).astype(int)
    sns.set_style('whitegrid')
    g = sns.catplot(
        data=df, x='SHAP', y='Data', col='Feature', hue='Hue', palette=[shap.plots.colors.blue_rgb, shap.plots.colors.red_rgb],
        kind='bar', orient='h', sharey=False, sharex=False, legend=False, col_wrap=2, 
        #col_order=['ED Diagnosis', 'Admission Specialty', 'Breathing Device', 'Admission Pathway', 'Presenting Complaint']
    )
    g.set_titles(template='{col_name}')
    g.fig.set_size_inches(8,10)
    g.fig.tight_layout()
    g.set_ylabels('')
    g.set_xlabels('Mean SHAP value')
    g.despine(left=True)
    g.fig.suptitle('Categorical Feature Interactions - GBDT (LightGBM)')
    g.fig.subplots_adjust(top=0.9)
    if save:
        plt.savefig(save, bbox_inches='tight', format='svg')

# %% [markdown]
# ### Scatter Plots

# %%
import shap
def comparison_scatter_plots(results, feature='Age', models=['LightGBM', 'L2Regression'], letters='ab', title=None, save=None):
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    feature_name = PRETTY_PRINT_FEATURE_NAMES[feature]

    for i, model in enumerate(models):
        explanations = results[(model, 'with_services')][-2]
        
        explanations = explanations[:,explanations.feature_names.index(feature)]
        df = pd.concat([pd.Series(explanations.values, name='Value'), pd.Series(explanations.data, name=feature)], axis=1)
        if feature == 'Obs_Temperature_Admission':
            df = df[df.Obs_Temperature_Admission > 34]
            df = df[df.Obs_Temperature_Admission < 42]

        df['Colour'] = df.Value.apply(lambda x: x < 0)
        sns.scatterplot(data=df, x=feature, y='Value', hue='Colour', ax=ax[i], legend=False, palette=[shap.plots.colors.red_rgb, shap.plots.colors.blue_rgb], linewidth=0, s=16)
        ax[i].set_title(f'({letters[i]}) {PRETTY_PRINT_ESTIMATORS[model]}')
        points = ax[i].collections[0]
        points.set_rasterized(True)

    ax[0].set_ylabel(f'SHAP value for \n {feature_name}')
    ax[1].set_ylabel('')
    ax[0].set_xlabel(feature_name)
    ax[1].set_xlabel(feature_name)

    if not title:
        title = f'Patient-individual SHAP values for {feature_name}'
    fig.suptitle(title)

    if save:
        fig.savefig(save, bbox_inches='tight', format='svg')


# %% [markdown]
# ## Regression Coefficinets

# %%
from transformer_experiment.utils.shallow_classifiers import get_calibrated_regression_coefficients

def get_onehot_columns():
    X, y = SalfordAdapter(SAL.loc[SAL_TRAIN_IDX]).xy(x=SalfordCombinations['with_services'], onehot_encoding=True)
    r = X.get_onehot_categorical_columns()
    for column in r.keys():
        r[column].append(f'{column}__NAN')
    return r

def plot_calibrated_regression_coefficients(
    model, columns, topn=60, figsize=(8, 12), pipeline_key=None, save=None,
):
    df = get_calibrated_regression_coefficients(model, columns, pipeline_key)
    regression_coefficient_sorted_barplot(df, topn=topn, save=save)

def get_logistic_regression_coefficients(model, estimator_name, onehot_cols):
    coef = get_calibrated_regression_coefficients(
        model, estimator_name
    ).set_index('Feature').Coefficient

    coef_num = coef[~coef.index.isin([__ for _ in onehot_cols.values() for __ in _])].sort_values()
    coef_cat = pd.DataFrame([dict(
        Feature=PRETTY_PRINT_FEATURE_NAMES[key],
        Value=_,
        Coef=coef.loc[_]    
    ) for key, val in onehot_cols.items() for _ in val if _ in coef.index])
    coef_cat.Value = coef_cat.Value.str.split('__').str[1]

    return coef_num, coef_cat

# %%
def categorical_coefficient_table(results, onehot_cols, estimators=['L1Regression', 'L2Regression', 'ElasticNetRegression'], feature_group='with_services'):
    r = pd.concat((
        get_logistic_regression_coefficients(
            results[(estimator, feature_group)][-1], estimator, onehot_cols
        )[1].assign(Estimator=PRETTY_PRINT_ESTIMATORS[estimator]).replace('NAN', 'Unknown')
    ) for estimator in estimators)

    top = r.copy()
    top.Coef = top.Coef.apply(abs)
    top = top.groupby(['Feature', 'Value']).mean().groupby('Feature').Coef.nlargest(5).droplevel(0).index
    return r.pivot(index=['Feature', 'Value'], columns='Estimator', values='Coef').loc[top].round(4)

# %%
def numerical_coefficient_table(results, onehot_cols, estimators=['L1Regression', 'L2Regression', 'ElasticNetRegression'], feature_group='with_services'):
    r = pd.concat((
        (
            get_logistic_regression_coefficients(results[(estimator, feature_group)][-1], estimator, onehot_cols)[0].rename(estimator)
        ) for estimator in estimators), axis=1
    ).round(4)

    r.index = map(PRETTY_PRINT_FEATURE_NAMES.get, r.index)
    r.columns = map(PRETTY_PRINT_ESTIMATORS.get, r.columns)
    return r


# %%
#print(numerical_logistic_coefficients_table.to_latex(bold_rows=True, column_format='lrrr',  formatters=[lambda x: f'${x}$' for _ in range(numerical_logistic_coefficients_table.shape[1])], escape=False))

# %% [markdown]
# ## Bias

# %% [markdown]
# ### Entropy Curves

# %%
import matplotlib.gridspec
from transformer_experiment.utils.plots import plot_entropy_curves
from aif360.sklearn.metrics import generalized_entropy_error, between_group_generalized_entropy_error

def entropy_comparison(y_preds, y_preds_transf, prot_attr=None, save=None, function=generalized_entropy_error, title='Fairness - Generalised Entropy Index'):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(10,10))
    gs = matplotlib.gridspec.GridSpec(4, 4, hspace=0.6, wspace=0.4)
    ax1 = plt.subplot(gs[:2, :2])
    ax2 = plt.subplot(gs[:2, 2:])
    ax3 = plt.subplot(gs[2:4, 1:3])

    y_true = Y_TRUES['Complete']
    baseline_news = {'NEWS2': SAL.loc[y_true.index].NEWS_Score_Admission}

    y_preds_l, y_preds_r, y_preds_d = (
        estimator_y_preds_across_feature_groups(y_preds, 'LightGBM') | baseline_news, 
        feature_group_y_preds_across_estimators(y_preds, 'with_services') | baseline_news,
        all_y_preds_by_estimator(y_preds_transf) | baseline_news
    )

    y_preds_r = {PRETTY_PRINT_ESTIMATORS[key]:value for key, value in y_preds_r.items()}

    plot_entropy_curves(
        y_true, y_preds_l, 
        function=function, prot_attr=prot_attr,
        ax=ax1, title='(a) GBDT (LightGBM)', 
        baseline_key='NEWS2', palette=sns.color_palette('muted')
    )
    plot_entropy_curves(
        y_true, y_preds_r, 
        function=function, prot_attr=prot_attr,
        ax=ax2, title='(b) All Models, Complete Feature Set', 
        baseline_key='NEWS2', palette=sns.color_palette('muted')
    )
    plot_entropy_curves(
        y_true, y_preds_d, 
        function=function, prot_attr=prot_attr,
        ax=ax3, title='(c) Language Modelling', 
        baseline_key='NEWS2', palette=sns.color_palette('muted'), ci=None
    )

    sns.move_legend(ax1, "upper left", bbox_to_anchor=(2.2, 1.1), frameon=False, title='(a) GBDT')
    sns.move_legend(ax2, "lower left", bbox_to_anchor=(1, -0.1), frameon=False, title='(b) All models')
    sns.move_legend(ax3, "center left", bbox_to_anchor=(1, .5), frameon=False, title='(c) Language Modelling')

    ax2.set_ylabel(None)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title)
    if save:
        plt.savefig(save, bbox_inches='tight', format='svg')


print('Starting entropy comparison')
# %%
SAL.loc[SAL_TEST_IDX, 'Ethnicity'] = SAL.loc[SAL_TEST_IDX, 'Ethnicity'].fillna("WHITE BRITISH")
entropy_comparison(
    RESULTS, TRANSF_RESULTS_SELECT, 
    function=between_group_generalized_entropy_error, 
    prot_attr=(
        SAL.loc[SAL_TEST_IDX].CriticalEvent.set_axis(
            SAL.loc[SAL_TEST_IDX, ['Female', 'Ethnicity']]
        ).index), 
    save=Notebook.IMAGE_DIR/'entropy_between.svg', 
    title='Between-Group Fairness - Generalised Entropy Index'
)
