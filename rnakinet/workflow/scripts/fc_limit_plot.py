import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from plot_helpers import setup_palette
from correlation_plot import correlation_plot
from scipy.stats import spearmanr, pearsonr

def calc_fc(df, col, conditions_count, controls_count):
    df['ctrl_avg'] = df[[f'{col}_ctrl_{i}' for i in range(controls_count)]].mean(axis=1)
    df['cond_avg'] = df[[f'{col}_cond_{i}' for i in range(conditions_count)]].mean(axis=1)
    #TODO not Fold Change - rename to someting better
    df['Pred_FC'] = df['cond_avg']/df['ctrl_avg']
    df['Relative modification increase (%)'] = ((df['cond_avg']/df['ctrl_avg'])-1)*100
    return df

def main(args):
    ctrl_paths = args.gene_level_preds_control
    cond_paths = args.gene_level_preds_condition
    deseq_path = args.deseq_output
    pred_col = args.pred_col
    target_col = args.target_col
    
    deseq_df = pd.read_csv(deseq_path, sep='\t', index_col=None).reset_index()
    deseq_df = deseq_df[~deseq_df['log2FoldChange'].isna()] #Dropping genes that dont contain fold change data
    
    cond_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in cond_paths]
    ctrl_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in ctrl_paths]
    
    
    for i,cond_df in enumerate(cond_dfs):
        cond_df.columns = [f'{col}_cond_{i}' if col != 'Gene stable ID' else col for col in cond_df.columns]
    
    for i,ctrl_df in enumerate(ctrl_dfs):
        ctrl_df.columns = [f'{col}_ctrl_{i}' if col != 'Gene stable ID' else col for col in ctrl_df.columns]
    
    all_dfs = cond_dfs+ctrl_dfs
    joined_df = all_dfs[0]
    for df in all_dfs[1:]:
        joined_df = joined_df.merge(df, on='Gene stable ID', how='outer')

    joined_df = joined_df.merge(deseq_df, left_on='Gene stable ID', right_on='index', how='right')
    joined_df = calc_fc(joined_df, pred_col, conditions_count = len(cond_dfs), controls_count = len(ctrl_dfs))
    
    # dropping all genes that dont appear in all experiments
    joined_df = joined_df[~joined_df.isna().any(axis=1)]
    
    #Filtering where Pred_FC is infinite or nan (after log division when some of the ratios are infinite)
    joined_df = joined_df.replace([np.inf, -np.inf], np.nan)
    joined_df = joined_df[~joined_df['Pred_FC'].isna()]
    
    joined_df['Pred_log2FoldChange'] = np.log2(joined_df['Pred_FC'])
    joined_df = joined_df[~joined_df['Pred_log2FoldChange'].isna()]
    joined_df['FC'] = 2**joined_df['log2FoldChange']
    
    
    spearman_corrs = []
    pearson_corrs = []
    
    og_cond_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in cond_paths]
    og_ctrl_dfs = [pd.read_csv(path, sep='\t', index_col=0) for path in ctrl_paths]
    cond_max = [df['reads'].max() for df in og_cond_dfs]
    ctrl_max = [df['reads'].max() for df in og_ctrl_dfs]
    all_max = max(cond_max+ctrl_max)
    limits = range(0, all_max)
    
    min_reads_to_plot = 30 
    for min_reads in limits:
        sub_joined_df = joined_df
        for i in range(len(cond_dfs)):
            sub_joined_df = sub_joined_df[sub_joined_df[f'reads_cond_{i}'] >= min_reads]
        for i in range(len(ctrl_dfs)):
            sub_joined_df = sub_joined_df[sub_joined_df[f'reads_ctrl_{i}'] >= min_reads]
    
        if(len(sub_joined_df) < min_reads_to_plot):
            break
            
        x = sub_joined_df['log2FoldChange'].values
        y = sub_joined_df['Relative modification increase (%)'].values
        spearman = spearmanr(x,y).statistic
        pearson = pearsonr(x,y).statistic
        
        spearman_corrs.append(spearman)
        pearson_corrs.append(pearson)
        
    fontsize=8
    
    palette = setup_palette()
    plt.figure(figsize=(1.5,1.5))
    
    plt.xlabel('Minimum read requirement', fontsize=fontsize)
    plt.ylabel('Correlation of expression fold change (log2)\n and relative modification increase (%)', fontsize=fontsize)
    
    plt.plot(limits[:len(spearman_corrs)], spearman_corrs, label='spearman', color=palette[0])
    plt.plot(limits[:len(pearson_corrs)], pearson_corrs, label='pearson', color=palette[1])
    
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    plt.legend(loc='lower right', fontsize=fontsize-2, frameon=False)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(args.output, bbox_inches = 'tight')
    
    # correlation_plot(joined_df, x_column='log2FoldChange',y_column='Relative modification increase (%)', x_label='Expression fold change (log2)\nHeat shock vs control', y_label='Relative modification increase (%)', output=args.output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make FC correlation plot from differential expression analysis data and predictions.")
    parser.add_argument("--gene-level-preds-control",nargs='+', type=str)
    parser.add_argument("--gene-level-preds-condition",nargs='+', type=str)
    parser.add_argument("--deseq-output", type=str)
    parser.add_argument("--pred-col", type=str)
    parser.add_argument("--target-col", type=str)
    
    parser.add_argument("--output", help="filename to save the plot")
    
    args = parser.parse_args()
    main(args)