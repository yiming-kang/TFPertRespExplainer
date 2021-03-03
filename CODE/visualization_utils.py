import numpy as np
import pandas as pd
import os.path
from glob import glob
import scipy.stats as ss
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score


COLORS = {
    'orange': '#f0593e', 
    'dark_red': '#7c2712', 
    'red': '#ed1d25',
    'yellow': '#ed9f22', 
    'light_green': '#67bec5', 
    'dark_green': '#018a84',
    'light_blue': '#00abe5', 
    'dark_blue': '#01526e', 
    'grey': '#a8a8a8'}

DINUCLEOTIDES = {
    'AA': 'AA/TT', 'AC': 'AC/GT', 'AG': 'AG/CT',
    'CA': 'CA/TG', 'CC': 'CC/GG', 'GA': 'GA/TC'
}


def parse_classifier_stats(dirpath, algorithm, feat_types, sys2com_dict=None):
    out_df = pd.DataFrame(
        columns=['tf', 'chance', 'feat_type', 'cv', 'auroc', 'auprc'])
    
    for feat_type in feat_types:
        print('... working on', feat_type)
        subdirs = glob('{}/{}/{}/*'.format(dirpath, feat_type, algorithm))
        
        for subdir in subdirs:
            tf = os.path.basename(subdir)
            filename = glob('{}/stats.csv*'.format(subdir))[0]
            stats_df = pd.read_csv(filename)
            filename = glob('{}/preds.csv*'.format(subdir))[0]
            preds_df = pd.read_csv(filename)
            stats_df['feat_type'] = feat_type
            
            if sys2com_dict is not None:
                tf_com = sys2com_dict[tf] if tf in sys2com_dict else tf
                stats_df['tf'] = '{} ({})'.format(tf, tf_com)
                stats_df['tf_com'] = tf_com
            else:
                stats_df['tf'] = tf
            
            stats_df['chance'] = np.sum(preds_df['label'] == 1) / preds_df.shape[0]
            out_df = out_df.append(stats_df, ignore_index=True)
    return out_df


def compare_model_stats(df, metric, comp_groups):
    stats_df = pd.DataFrame(columns=['tf', 'comp_group', 'p_score'])
    
    for tf, df2 in df.groupby('tf'):
        for (f1, f2) in comp_groups:
            x1 = df2.loc[df2['feat_type'] == f1, metric]
            x2 = df2.loc[df2['feat_type'] == f2, metric]
            _, p = ss.ttest_rel(x1, x2)
            sign = '+' if np.median(x2) > np.median(x1) else '-'
           
            stats_row = pd.Series({
                'tf': tf, 
                'comp_group': '{} vs {}'.format(f1, f2), 
                'p_score': -np.log10(p),
                'sign': sign})
            stats_df = stats_df.append(stats_row, ignore_index=True)
    return stats_df


def get_feature_indices(df, tf_name, organism):
    if organism == 'yeast':
        feat_dict = {
            'tf_binding:{}'.format(tf_name): 'TF binding', 
            'histone_modifications:h3k27ac_tp1_0_merged': 'H3K27ac',
            'histone_modifications:h3k36me3_tp1_0_merged': 'H3K36me3',
            'histone_modifications:h3k4me3_tp1_0_merged': 'H3K4me3',
            'histone_modifications:h3k4me_tp1_0_merged': 'H3K4me1',
            'histone_modifications:h3k79me_tp1_0_merged': 'H3K79me1',
            'histone_modifications:h4k16ac_tp1_0_merged': 'H4K16ac',
            'chromatin_accessibility:BY4741_ypd_osm_0min.occ': 'Chrom acc',
            'gene_expression:{}'.format(tf_name): 'GEX level', 
            'gene_expression:variation': 'GEX var',
            'dna_sequence:nt_freq_agg': 'Dinucleotides'}
    elif organism == 'human':
        feat_dict = {
            'tf_binding:{}'.format(tf_name): 'TF binding', 
            'histone_modifications:K562_H3K27ac': 'H3K27ac',
            'histone_modifications:K562_H3K27me3': 'H3K27me3',
            'histone_modifications:K562_H3K36me3': 'H3K36me3',
            'histone_modifications:K562_H3K4me1': 'H3K4me1',
            'histone_modifications:K562_H3K4me3': 'H3K4me3',
            'histone_modifications:K562_H3K9me3': 'H3K9me3',
            'chromatin_accessibility:K562_atac': 'Chrom acc',
            'gene_expression:median_level': 'GEX level', 
            'gene_expression:variation': 'GEX var',
            'dna_sequence:nt_freq_agg': 'DNA sequence'}

    idx_df = pd.DataFrame()
    for _, row in df.iterrows():
        if row['feat_type'] == 'dna_sequence_nt_freq':
            type_name = 'dna_sequence:nt_freq_agg'
        else:
            type_name = row['feat_type'] + ':' + row['feat_name']
        type_name2 = feat_dict[type_name]
        for i in range(row['start'], row['end']):
            idx_df = idx_df.append(pd.Series({'feat_type_name': type_name2, 'feat_idx': i}), ignore_index=True)
    return idx_df


def calculate_resp_and_unresp_signed_shap_sum(tf_dir, tf_name, organism):
    shap_df = pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(tf_dir))
    feats_df = pd.read_csv('{}/feats.csv.gz'.format(tf_dir), names=['feat_type', 'feat_name', 'start', 'end'])
    preds_df = pd.read_csv('{}/preds.csv.gz'.format(tf_dir))
    feat_idx_df = get_feature_indices(feats_df, tf_name, organism)

    ## Parse out shap+ and shap- values
    shap_df = shap_df.rename(columns={'feat': 'shap'})
    shap_df = shap_df.merge(preds_df[['gene', 'label']], how='left', on='gene')
    shap_df['shap+'] = [x if x > 0 else 0 for x in shap_df['shap']]
    shap_df['shap-'] = [x if x < 0 else 0 for x in shap_df['shap']]

    ## Sum across reg region for each feature and each gene, and then take 
    ## the mean among responsive targets and repeat for non-responsive targets.
    shap_df = shap_df.merge(feat_idx_df[['feat_type_name', 'feat_idx']], on='feat_idx')
    sum_shap = shap_df.groupby(['gene', 'label', 'feat_type_name'])[['shap+', 'shap-']].sum().reset_index()
    sum_shap = sum_shap.groupby(['label', 'feat_type_name'])[['shap+', 'shap-']].mean().reset_index()
    sum_shap['label_name'] = ['Responsive' if x == 1 else 'Non-responsive' for x in sum_shap['label']]
    sum_shap['label_name'] = pd.Categorical(
        sum_shap['label_name'], ordered=True, categories=['Responsive', 'Non-responsive'])

    sum_shap_pos = sum_shap[['label_name', 'feat_type_name', 'shap+']].copy().rename(columns={'shap+': 'shap'})
    sum_shap_pos['shap_dir'] = 'SHAP > 0'
    sum_shap_neg = sum_shap[['label_name', 'feat_type_name', 'shap-']].copy().rename(columns={'shap-': 'shap'})
    sum_shap_neg['shap_dir'] = 'SHAP < 0'
    
    sum_signed_shap = pd.concat([sum_shap_pos, sum_shap_neg])
    sum_signed_shap['shap_dir'] = pd.Categorical(sum_signed_shap['shap_dir'], categories=['SHAP > 0', 'SHAP < 0'])
    return sum_signed_shap


def get_best_yeast_model(data_dirs, tf_name):
    tf1_dir = '{}/{}'.format(data_dirs[0], tf_name)
    tf2_dir = '{}/{}'.format(data_dirs[1], tf_name)
    if (not os.path.exists(tf1_dir)) and (not os.path.exists(tf2_dir)):
        return None
    elif (os.path.exists(tf1_dir)) and (os.path.exists(tf2_dir)):
        acc1 = pd.read_csv('{}/stats.csv.gz'.format(tf1_dir))['auprc'].median()
        acc2 = pd.read_csv('{}/stats.csv.gz'.format(tf2_dir))['auprc'].median()
        tf_dir = tf1_dir if acc1 >= acc2 else tf2_dir
        is_cc = True if acc1 >= acc2 else False
        acc = max(acc1, acc2)
    else:
        tf_dir = tf1_dir if os.path.exists(tf1_dir) else tf2_dir
        is_cc = True if os.path.exists(tf1_dir) else False
        acc = pd.read_csv('{}/stats.csv.gz'.format(tf_dir))['auprc'].median()
    return (tf_dir, is_cc, acc)
    

def calculate_shap_net_influence(df):
    df2 = pd.DataFrame()
    for (label_name, feat_type_name, tf), subdf in df.groupby(['label_name', 'feat_type_name', 'tf']):
        shap_diff = subdf.loc[subdf['shap_dir'] == 'SHAP > 0', 'shap'].iloc[0] - \
                    np.abs(subdf.loc[subdf['shap_dir'] == 'SHAP < 0', 'shap'].iloc[0])
        df2 = df2.append(pd.Series({
            'label_name': label_name, 'feat_type_name': feat_type_name,
            'tf': tf, 'acc': subdf['acc'].iloc[0], 'shap_diff': shap_diff
            }), ignore_index=True)
    return df2


def calculate_tf_binding_shap(resp_df, tf_dir, tf_name, bound_targets, tfb_params):
    shap_df = pd.read_csv('{}/feat_shap_wbg.csv.gz'.format(tf_dir))
    preds_df = pd.read_csv('{}/preds.csv.gz'.format(tf_dir))
    
    feats_df = pd.read_csv('{}/feats.csv.gz'.format(tf_dir), names=['feat_type', 'feat_name', 'start', 'end'])
    feat_mtx = np.loadtxt('{}/feat_mtx.csv.gz'.format(tf_dir), delimiter=',')
    genes = np.loadtxt('{}/genes.csv.gz'.format(tf_dir), dtype=str)
    feat_mtx = pd.DataFrame(data=feat_mtx, index=genes)
    
    annot_genes_df = annotate_resp_type(resp_df, feat_mtx, feats_df, tf_name, bound_targets=bound_targets)
    
    ## Pare down to the TF's binding feature
    feat_type, feat_shift, bin_width = tfb_params
    feat_row = feats_df[(feats_df['feat_type'] == feat_type) & (feats_df['feat_name'] == tf_name)].iloc[0]
    
    comb_vals_df = pd.DataFrame()
    for feat_idx in range(feat_row['start'], feat_row['end']):
        comb_vals = preds_df.merge(shap_df.loc[shap_df['feat_idx'] == feat_idx], on='gene')
        comb_vals = comb_vals.merge(feat_mtx[feat_idx], left_on='gene', right_index=True)
        comb_vals = comb_vals.rename(columns={'feat': 'shap', feat_idx: 'input'})
        comb_vals['label'] = comb_vals['label'].astype(int).astype(str)
        comb_vals['coord'] = (feat_idx - feat_row['start']) * bin_width - feat_shift
        comb_vals = comb_vals.merge(annot_genes_df, how='left', left_on='gene', right_index=True)
        comb_vals_df = comb_vals_df.append(comb_vals, ignore_index=True)
    return comb_vals_df

        
def get_callingcards_bound_targets(cc_dir, tf_name, gene_list):
    cc_thresh = 10**-3
    tfb_sig = pd.read_csv('{}/{}.sig_prom.txt'.format(cc_dir, tf_name), sep='\t')
    bound_targets = tfb_sig.loc[tfb_sig['Poisson pvalue'] < cc_thresh, 'Systematic Name']
    return np.intersect1d(bound_targets, gene_list)


def annotate_resp_type(resp_df, feat_mtx, feats_df, tf_name, bound_targets=None):
    tfb_idx = feats_df.loc[(feats_df['feat_type'] == 'tf_binding') & (feats_df['feat_name'] == tf_name), ['start', 'end']].iloc[0]
    tfb_sum = feat_mtx[range(tfb_idx['start'], tfb_idx['end'])].sum(axis=1).to_frame().rename(columns={0: 'tfb'})
    
    resp_df = resp_df[tf_name].to_frame().rename(columns={tf_name: 'resp'})
    resp_df = resp_df.merge(tfb_sum, left_on='GeneName', right_index=True)
    
    resp_df['is_bound'] = 'Unbound'
    if bound_targets is None:
        resp_df.loc[resp_df['tfb'] > 0, 'is_bound'] = 'Bound'
    else:
        resp_df.loc[bound_targets, 'is_bound'] = 'Bound'

    resp_df['resp_dir'] = 'Non-responsive'
    resp_df.loc[resp_df['resp'] != 0, 'resp_dir'] = 'Responsive'
    return resp_df[['is_bound', 'resp_dir']]
