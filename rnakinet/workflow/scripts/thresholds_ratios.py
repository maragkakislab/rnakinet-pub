from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import argparse
from plot_helpers import plot_and_save, parse_plotting_args
import seaborn as sns
from plot_helpers import setup_palette
import matplotlib as mpl
import math
from matplotlib.ticker import MaxNLocator

def plot_thresholds(pos_preds, neg_preds, pos_name, neg_name):
    thresholds = np.arange(0,1.1,0.1)
    balance_ratios = np.linspace(0.0, 1.0, 11)
    
    heatmap_data = np.zeros((len(balance_ratios), len(thresholds)))
    # For pos-total ratio
    total_preds = len(pos_preds)+len(neg_preds)
    
    for i,balance_ratio in enumerate(balance_ratios):
        current_pos_ratio = len(pos_preds)/total_preds
        current_neg_ratio = len(neg_preds)/total_preds

        wanted_pos_ratio = balance_ratio
        wanted_neg_ratio = 1-balance_ratio
        
        #downsample only
        if(current_pos_ratio >= wanted_pos_ratio):
            pos_samples_needed = (wanted_pos_ratio * len(neg_preds))/(1-wanted_pos_ratio)
            balanced_pos_preds = pos_preds[:int(pos_samples_needed)]
            balanced_neg_preds = neg_preds
        if(current_neg_ratio >= wanted_neg_ratio):
            neg_samples_needed = (wanted_neg_ratio * len(pos_preds))/(1-wanted_neg_ratio)
            balanced_pos_preds = pos_preds
            balanced_neg_preds = neg_preds[:int(neg_samples_needed)]
        
        total_balanced_preds = len(balanced_pos_preds)+len(balanced_neg_preds)
        current_pos_ratio = len(balanced_pos_preds)/total_balanced_preds
        current_neg_ratio = len(balanced_neg_preds)/total_balanced_preds
        assert(math.isclose(current_pos_ratio, wanted_pos_ratio, rel_tol=1e-3)),f'{current_pos_ratio}, {wanted_pos_ratio}'
        assert(math.isclose(current_neg_ratio, wanted_neg_ratio, rel_tol=1e-3)),f'{current_neg_ratio}, {wanted_neg_ratio}'
        
        accs = [np.mean(np.concatenate([balanced_neg_preds<=t,balanced_pos_preds > t])) for t in thresholds]
        heatmap_data[i, :] = accs
        
        
    norm = matplotlib.colors.Normalize(0,1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [setup_palette()[1], setup_palette()[0]])
    ax = sns.heatmap(
        heatmap_data, 
        xticklabels=np.round(thresholds, 2), 
        yticklabels=[f'{br:.1f}' for br in balance_ratios],
        cmap=cmap, 
        norm=norm,
        annot=False,
        # vmin=0, 
        # vmax=1,
    )

    mpl.rc('font',family='Arial')
    fontsize=8
    plt.xlabel('Threshold', fontsize=fontsize)
    plt.ylabel('Ratio', fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    sns.set_style('whitegrid')
    sns.despine()
    plt.legend(loc='lower right', frameon=False, fontsize=fontsize-2)
    
    
def get_threshold_callback(args):
    def threshold_line_callback():
        palette = setup_palette()
        color = palette[4]
        # plt.axvline(x=args.chosen_threshold, color=color, linestyle='--')
    return threshold_line_callback
    

def main(args):
    plot_and_save(args, plot_thresholds, [get_threshold_callback(args)])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a balanced accuracy plot for various thresholds')
    parser = parse_plotting_args(parser)
    args = parser.parse_args()
    main(args)