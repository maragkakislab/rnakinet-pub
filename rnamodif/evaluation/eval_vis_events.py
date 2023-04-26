from fast_ctc_decode import beam_search, viterbi_search
import torch
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from difflib import SequenceMatcher
from Bio import Align
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def get_mismatches(batches, example_batch=0):
    mismatches = {}
    cm = np.zeros((2,2))
    diversity_percentages = {'A => A':[], 'A => X':[], 'X => X':[], 'X => A':[]}
    example_alignments = {0:{0:None,1:None},1:{0:None,1:None}}
    
    
    for batch_num, batch in enumerate(tqdm(batches)):
        global_pred = torch.softmax(batch[0].float(), dim=2).detach().cpu().numpy()
        for el in range(batch[0].size()[1]):
            el_pred = global_pred[:,el,:] 
            seq, path = beam_search(el_pred, alphabet='NACGTX')
            # aligner = Align.PairwiseAligner()
            # aligner.mode = 'global'
            # aligner.match_score = 2   # score for matching characters
            # aligner.mismatch_score = -1   # score for mismatched characters
            # aligner.open_gap_score = -5 
            # aligner.extend_gap_score = -2

            seq1 = batch[1][el]
            seq2 = seq
            
            diversity_counts = {'A => A':0, 'A => X':0, 'X => X':0, 'X => A':0}
            
            
            
            for alignment in list(pairwise2.align.globalms(seq1, seq2, 2, -1, -5, -2, one_alignment_only=True))[:1]: #only first alignment
            # for alignment in list(aligner.align(seq1, seq2))[:1]: #only first alignment
                
                label_sequence, predicted_sequence = alignment[0], alignment[1]
                if('X' in label_sequence):
                    label_index = 1
                else: 
                    label_index = 0
                if('X' in predicted_sequence):
                    predicted_index = 1
                else:
                    predicted_index = 0
                
                if(not example_alignments[label_index][predicted_index]):
                    if(batch_num == example_batch):
                        example_alignments[label_index][predicted_index] = alignment
                    
                    
                cm[label_index][predicted_index]+=1
                
                for lab, pred in zip(alignment[0], alignment[1]):
                    s = f'{lab} => {pred}'
                    
                    spots = ['A','X']                    
                    if(pred in spots and lab in spots):
                        diversity_counts[s]+=1
                    
                    if((pred == 'A' or pred == 'X') and pred==lab):
                        if(s not in mismatches.keys()):
                            mismatches[s] = 1
                        else:
                            mismatches[s]+=1
                    if pred!=lab:
                        if(s not in mismatches.keys()):
                            mismatches[s] = 1
                        else:
                            mismatches[s]+=1
            
            if(sum(diversity_counts.values())>0):
                for k,v in diversity_counts.items():
                    diversity_percentages[k].append(diversity_counts[k]/sum(diversity_counts.values()))
    
    # print(diversity_percentages)
    return dict(sorted(mismatches.items(), key=lambda item: item[1], reverse=True)), diversity_percentages, cm, example_alignments


def plot_diversity(div):
    fig, axs = plt.subplots(2,2, sharey=True)
    for i,(k,v) in enumerate(div.items()):
        x_idx = i%2
        y_idx = i//2
        axs[x_idx,y_idx].hist(v)
        axs[x_idx,y_idx].set_title(k)
    fig.tight_layout()
    plt.show()
    
def plot_cm(cm):
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['A','X'], yticklabels=['A','X'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix (window-wise label)')
    plt.show()
    
    
def plot_alignments(alignments):
    for lab in range(2):
        for pred in range(2):
            print(f'label:{lab}, prediction:{pred}')
            # print(alignments[lab][pred])
            print(pairwise2.format_alignment(*alignments[lab][pred]))