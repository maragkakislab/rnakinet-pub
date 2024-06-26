import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from plot_helpers import setup_palette
from scipy.stats import spearmanr, pearsonr

def main(args):
    table = pd.read_csv(args.table, sep='\t')
    correlation_plot(table, x_column=args.x_column,y_column=args.y_column, x_label=args.x_label,y_label=args.y_label, output=args.output, share_axes=args.share_axes)

def correlation_plot(df, x_column, y_column, x_label, y_label, output, share_axes):
    plt.figure(figsize=(1.5,1.5))
    palette = setup_palette()
    sns.regplot(data=df, x=x_column, y=y_column, 
            scatter_kws={'alpha':0.6, 's':7, 'color':palette[0]}, 
            line_kws={"color": palette[1], "lw": 2},  
    )
    
    x = df[x_column].values
    y = df[y_column].values
    
    if share_axes:
        max_range = max((x.max()-x.min()),y.max()-y.min())
        padding = max_range*0.05
        common_range = [
            min(x.min(), y.min())-padding, 
            max(x.max(), y.max())+padding
        ]
        plt.xlim(common_range)
        plt.ylim(common_range)
        
        y_lim = plt.gca().get_ylim()
        y_ticks = plt.gca().get_yticks()
        y_ticks_visible = y_ticks[(y_ticks >= y_lim[0]) & (y_ticks <= y_lim[1])]
        
        plt.gca().set_xticks(y_ticks_visible)
    
    fontsize=8
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    
    spearman = spearmanr(x,y).statistic
    pearson = pearsonr(x,y).statistic
    
    plt.text(0.1, 0.95, f'r={pearson:.2f} ρ={spearman:.2f}', fontsize=fontsize-2, transform=plt.gca().transAxes, verticalalignment='top')
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    sns.set_style('whitegrid')
    sns.despine()
    
    plt.savefig(output, bbox_inches='tight') 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=str, required=True)
    parser.add_argument('--x-column', type=str, required=True)
    parser.add_argument('--y-column', type=str, required=True)
    parser.add_argument('--x-label', type=str, required=True)
    parser.add_argument('--y-label', type=str, required=True)
    parser.add_argument('--share-axes', action=argparse.BooleanOptionalAction)
    parser.add_argument('--output', type=str, help='Path to the output plot.')
    
    args = parser.parse_args()
    main(args)
    
    
    