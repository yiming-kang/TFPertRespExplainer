import sys
import configparser
import logging.config
from copy import deepcopy

import numpy as np
import pandas as pd
import multiprocess as mp
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score
import xgboost as xgb
import shap

from modeling_utils import *

## Intialize logger
logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

## Load default configuration
config = configparser.ConfigParser()
config.read('config.ini')
RAND_NUM = int(config['DEFAULT']['rand_num'])
np.random.seed(RAND_NUM)
MAX_RECURSION = int(config['DEFAULT']['max_recursion'])
sys.setrecursionlimit(MAX_RECURSION)
BG_GENE_NUM = 500


class TFPRExplainer:
    def __init__(self, feat_mtx, features, label_df):
        self.X = feat_mtx
        self.y = label_df
        self.feats = features
        self.genes = label_df.index.values
        self.k_folds = 10

    def cross_validate(self, is_regressor=False):
        """Cross valdiate a classifier or regressor using multiprocessing.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}

            if is_regressor:
                kfolds = KFold(
                    n_splits=self.k_folds, shuffle=True, random_state=RAND_NUM)
            else:
                kfolds = StratifiedKFold(
                    n_splits=self.k_folds, shuffle=True, random_state=RAND_NUM)

            for k, (tr_idx, te_idx) in enumerate(kfolds.split(self.X, self.y)):

                # TODO: only run on the first fold
                if k != 3:
                    continue

                y_tr, y_te = self.y[tr_idx], self.y[te_idx]
                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                mp_results[k] = pool.apply_async(
                    train_and_predict,
                    args=(k, (X_tr, y_tr), (X_te, y_te), is_regressor,))

            self.cv_results = compile_mp_results(mp_results)

    def explain(self):
        """Use SHAP values to features' contributions to predict the 
        responsiveness of a gene.
        """
        with mp.Pool(processes=self.k_folds) as pool:
            mp_results = {}

            for k, y_te in enumerate(self.cv_results['preds']):
                te_genes = y_te['gene'].values
                te_idx = [np.where(self.genes == g)[0][0] for g in te_genes]
                
                tr_idx = sorted(set(range(len(self.genes))) - set(te_idx))
                logger.info('Explaining {} genes in fold {}'.format(len(te_idx), k))

                X_tr, X_te = self.X[tr_idx], self.X[te_idx]
                X_tr, X_te = standardize_feat_mtx(X_tr, X_te, 'zscore')

                bg_idx = np.random.choice(
                    range(X_tr.shape[0]), BG_GENE_NUM, replace=False)
                mp_results[k] = pool.apply_async(
                    calculate_tree_shap,
                    args=(
                        self.cv_results['models'][k], 
                        X_te, te_genes, X_tr[bg_idx],))
            
            self.shap_vals = [mp_results[k].get() for k in sorted(mp_results.keys())]

    def save(self, dirpath):
        """Save output data.
        """
        # TODO: turn back on SHAP explainer
        # pd.concat(self.cv_results['preds']).to_csv(
        #     '{}/preds.csv.gz'.format(dirpath), 
        #     index=False, compression='gzip')

        pd.concat(self.cv_results['stats']).to_csv(
            '{}/stats.csv.gz'.format(dirpath), 
            index=False, compression='gzip')

        # np.savetxt(
        #     '{}/feats.csv.gz'.format(dirpath), np.array(self.feats),
        #     fmt='%s', delimiter=',')

        # np.savetxt(
        #     '{}/genes.csv.gz'.format(dirpath), self.genes,
        #     fmt='%s', delimiter=',')
    
        # np.savetxt(
        #     '{}/feat_mtx.csv.gz'.format(dirpath), self.X,
        #     fmt='%.8f', delimiter=',')

        # for k, df in enumerate(self.shap_vals):
        #     df['cv'] = k
        #     self.shap_vals[k] = df
        # pd.concat(self.shap_vals).to_csv(
        #     '{}/feat_shap_wbg.csv.gz'.format(dirpath),
        #     index=False, compression='gzip')


def train_and_predict(k, D_tr, D_te, is_regressor):
    """Train classifier and predict gene responses. 
    """
    logger.info('Cross validating fold {}'.format(k))

    X_tr, y_tr = D_tr
    X_te, y_te = D_te
    n_te_samples, n_feats = X_te.shape

    if is_regressor:
        ## Train regressor and test
        model = train_regressor(X_tr, y_tr)

        y_pred = model.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        stats_df = pd.DataFrame({'cv': [k], 'r2': [r2]})

        logger.info('Cross-validation R2={:.3f} in fold {}'.format(r2, k))

    else:        
        ## Train classifier and test
        model = train_classifier(X_tr, y_tr)

        y_pred = pd.DataFrame(
            data=model.predict_proba(X_te), 
            columns=model.classes_)[1].values
        auprc = average_precision_score(y_te, y_pred)
        auroc = roc_auc_score(y_te, y_pred)
        stats_df = pd.DataFrame({'cv': [k], 'auroc': [auroc], 'auprc': [auprc]})
        
        logger.info('Cross-validation AUPRC={:.3f} in fold {}'.format(auprc, k))

    return {
        'preds': pd.DataFrame({
            'gene': y_te.index.values, 'cv': [k] * n_te_samples, 
            'label': y_te.values, 'pred': y_pred}),
        'stats': stats_df,
        'models': model}


def train_classifier(X, y):
    """Train a XGBoost classifier.
    """
    # Tuning Notes:
    # - Resonable class weight helps (not as high as neg/pos ratio)
    # - Gamma helps as regularizer
    # - Number of features per tree helps  
    # - Subsample helps

    # neg_pos_ratio = sum(y == 0) / sum(y == 1)
    # neg_pos_ratio = 5
    neg_pos_ratio = 1

    # model = xgb.XGBClassifier(
    #     n_estimators=500,
    #     learning_rate=.01,
    #     booster='gbtree',
    #     scale_pos_weight=neg_pos_ratio,
    #     gamma=5,
    #     colsample_bytree=.8,
    #     subsample=.8,
    #     n_jobs=-1,
    #     random_state=RAND_NUM
    # )

    opt_hps = tune_hyparams(X, y)
    print('@----- Opitmized hyparams -----@\n', opt_hps, '\n')

    model = xgb.XGBClassifier(
        n_estimators=500,
        booster='gbtree',
        scale_pos_weight=neg_pos_ratio,
        n_jobs=-1,
        random_state=RAND_NUM,
        learning_rate=opt_hps['learning_rate'],
        gamma=opt_hps['gamma'],
        colsample_bytree=opt_hps['colsample_bytree'],
        subsample=opt_hps['subsample']
    )

    model.fit(X, y)
    return model


def train_regressor(X, y):
    """Train a XGBoost regressor.
    """
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=.01,
        objective='reg:squarederror',
        booster='gbtree',
        n_jobs=-1,
        random_state=RAND_NUM
    )
    model.fit(X, y)
    return model


def tune_hyparams(X, y):
    """Returns a dictionary of optimized hyperparameters.
    """
    from sklearn.model_selection import cross_val_predict
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    ## Define objective for optimization
    def model_objective(params):
        model = xgb.XGBClassifier(
            n_estimators=500,
            booster='gbtree',
            scale_pos_weight=1,
            n_jobs=-1,
            **params)
        y_pred = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
        loss = -average_precision_score(y, y_pred)  ## AuPRC as loss
        return {'loss': loss, 'status': STATUS_OK}

    ## Define search space
    hyparams = {
        # 'learning_rate': hp.loguniform('learning_rate', .001, .1),
        'learning_rate': hp.uniform('learning_rate', .005, .02),
        'gamma': hp.uniform('gamma', 0, 10),
        'colsample_bytree': hp.uniform('colsample_bytree', .5, 1.),
        'subsample': hp.uniform('subsample', .5, 1.)
    }

    # eps = .00001
    # hyparams = {
    #     'learning_rate': hp.uniform('learning_rate', .01 - eps, .01 + eps),
    #     'gamma': hp.uniform('gamma', 5 - eps, 5 + eps),
    #     'colsample_bytree': hp.uniform('colsample_bytree', .8 - eps, .8 + eps),
    #     'subsample': hp.uniform('subsample', .8 - eps, .8 + eps)
    # }
    space = hp.choice('model', [hyparams])

    ## Optimize hyparams
    trials = Trials()
    opt_hps = fmin(model_objective, space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(trials.trials)
    return opt_hps


def calculate_tree_shap(model, X, genes, X_bg):
    """Calcualte SHAP values for tree-based model.
    """
    n_genes, n_feats = X.shape
    
    ## Calculate SHAP values
    explainer = shap.TreeExplainer(model, X_bg)
    shap_mtx = explainer.shap_values(X, approximate=False, check_additivity=False)
    
    ## Convert wide to long format
    shap_df = pd.DataFrame(
        data=shap_mtx,
        index=genes,
        columns=['feat' + str(i) for i in range(n_feats)])
    shap_df.index.name = 'gene'
    shap_df = shap_df.reset_index()
    shap_df = pd.wide_to_long(shap_df, 'feat', 'gene', 'feat_idx').reset_index()
    return shap_df
