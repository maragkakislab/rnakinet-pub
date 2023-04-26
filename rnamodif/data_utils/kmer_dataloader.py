import random
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import numpy as np
from taiyaki.mapped_signal_files import HDF5Reader
# from rnamodif.data_utils.workers import worker_init_event_batch_fn
from rnamodif.data_utils.generators import uniform_gen, alternating_gen
from tqdm import tqdm

class MyKmerMixedDataset(IterableDataset):
    def __init__(self, pos_path, neg_path, max_kmer_len, batch_limit=None, total_limit=None, dont_pad=False, limit_avg_base=True):
        self.pos_dset = MyKmerDataset(pos_path, label=1, batch_limit=batch_limit, max_kmer_len=max_kmer_len, dont_pad=dont_pad, limit_avg_base=limit_avg_base)
        self.neg_dset = MyKmerDataset(neg_path, label=0, batch_limit=batch_limit, max_kmer_len=max_kmer_len, dont_pad=dont_pad, limit_avg_base=limit_avg_base)
        self.total_limit = total_limit
        
    def get_stream(self):
        print('pos kmer', self.pos_dset.kmer)
        print('neg kmer', self.neg_dset.kmer)
        pos_dset = iter(self.pos_dset)
        neg_dset = iter(self.neg_dset)
        if(self.total_limit):
            for _ in range(self.total_limit):
                yield next(pos_dset)
                yield next(neg_dset)
        else:
            while True:
                yield next(pos_dset)
                yield next(neg_dset)
    
    def __iter__(self):
        return self.get_stream()
    

class MyKmerDataset(IterableDataset):
    def __init__(self, hdf5_reader_path, label, max_kmer_len, batch_limit=None, dont_pad=False, limit_avg_base=True):
        self.reader = HDF5Reader(hdf5_reader_path)
        self.exp = Path(hdf5_reader_path).stem
        nums = self.reader.get_alphabet_information().collapse_labels
        letters = list(self.reader.get_alphabet_information().collapse_alphabet)
        self.vocab_map = {x:y for (x,y) in zip(letters,nums)}
        self.vocab_map_reversed = {v:k for k,v in self.vocab_map.items()}
        self.label = label
        self.max_kmer_len = max_kmer_len
        self.dont_pad = dont_pad
        self.limit_avg_base = limit_avg_base
        dataset = {}
        #TODO limit?
        counts = {}
        # lengths = []
        print(self.reader.batch_names)
        if(batch_limit):
            batch_names = np.array(self.reader.batch_names)[batch_limit]
        else:
            batch_names = self.reader.batch_names
            
        for batch_name in batch_names:
            print('processing', batch_name)
            reads_batch = self.reader._load_reads_batch(batch_name)
            for readid, mapping in tqdm(reads_batch.items()): 
                # signal = self.process_signal_mapping(mapping)
                letter_reference = [self.vocab_map_reversed[num] for num in mapping.Reference]
                ranges = [(mapping.Ref_to_signal[i], mapping.Ref_to_signal[i+1]) for i in range(len(mapping.Ref_to_signal)-1)]
                letter_to_ref = list(zip(letter_reference, ranges))
                for i in range(len(letter_to_ref)-4): #TODO do 7 instead of 5? (2 more kmers around A)
                    seq = ''
                    bases_lengths = []
                    for j in range(5):
                        letter, interval = letter_to_ref[i+j]
                        bases_lengths.append(interval[1] - interval[0])
                        if(j == 0):
                            start = interval[0]
                        if(j == 4):
                            end = interval[1]
                        seq+=letter
                       
                    bl = np.array(bases_lengths)
                    # if(((bl < 20) | (bl > 70)).any()):
                    if(self.limit_avg_base):
                        if((bl.mean() < 20) or (bl.mean() > 70)):
                            continue
                    
                    # signal_length = end-start 
                    # lengths.append(signal_length)
                    if(seq not in counts.keys()):
                        counts[seq] = 1
                    else:
                        counts[seq]+=1
                    #Limiting to specific kmer
                    # if(seq==kmer and signal_length<=max_kmer_len):
                    # if(seq==kmer):
                    #     if(seq not in dataset.keys()):
                    #         dataset[seq] = [(start,end, mapping)]
                    #     else:
                    #         dataset[seq].append((start,end, mapping))
                    if(seq not in dataset.keys()):
                        dataset[seq] = [(start,end, mapping)]
                    else:
                        dataset[seq].append((start,end, mapping)) 
                            
        # print(dict(sorted(counts.items(), key=lambda item: item[1])))
        self.dataset = dataset
        self.kmer = None
        self.shortcuts = None
        # self.shortcuts = self.dataset[kmer] #TODO parametrize
        self.counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        
        
        # self.lengths = lengths
        self.len = None
        print('dset size', self.len)
        # print('95 perc shorter than', np.percentile(lengths, q=95))
        
    def change_kmer(self, kmer):
        self.shortcuts = self.dataset[kmer]
        self.len = len(self.shortcuts)
        self.kmer = kmer
        
    def process_signal_mapping(self, mapping):
        #Taken from RODAN code
        signal = (mapping.Dacs + mapping.offset) * mapping.range / mapping.digitisation

        med = np.median(signal)
        mad = mapping.offset * np.median(abs(signal-med))
        signal = (signal - mapping.shift_frompA) / mapping.scale_frompA
        return signal

    
    def get_stream(self):
        if(not self.shortcuts):
            raise Exception('Initialize kmer')
        while True:
            idx = torch.randint(0, self.len, (1,))
            start, end, mapping = self.shortcuts[idx]
            arr = self.process_signal_mapping(mapping)[start:end]
            
            if(not self.dont_pad): #can be max_kmer_len != None
                arr = np.pad(arr, (0, self.max_kmer_len-len(arr)), mode='constant', constant_values=0)
                
            # arr = moving_average(arr, window_size=10)
            
            arr = np.array(arr, dtype=np.float32)
            yield arr, np.array([self.label], dtype=np.float32)
            # yield np.array([np.mean(arr),np.std(arr),len(arr)], dtype=np.float32), np.array([self.label], dtype=np.float32)
            
    
    def __iter__(self):
        return self.get_stream()
    
#     def __len__(self):
#         return len(self.shortcuts)
    
#     def __getitem__(self, idx):
#         start, end, mapping = self.shortcuts[idx]
#         arr = self.process_signal_mapping(mapping)[start:end]
#         padded_arr = np.pad(arr, (0, 600-len(arr)), mode='constant', constant_values=0)
#         return padded_arr
        

def moving_average(x, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(x, window, 'same')