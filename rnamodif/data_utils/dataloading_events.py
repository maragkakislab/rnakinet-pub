import random
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import numpy as np
from taiyaki.mapped_signal_files import HDF5Reader
from rnamodif.data_utils.generators import uniform_gen, alternating_gen


class event_datamodule(pl.LightningDataModule):
    def __init__(self, readers_split_config, window, valid_limit, batch_size=256, workers=1, simple_loader=False):
        super().__init__()
        self.config = readers_split_config
        self.window = window
        self.workers=workers
        self.batch_size = batch_size
        self.vocab_map = {'A':0,'C':1,'G':2,'T':3}
        self.valid_limit = valid_limit
        self.simple_loader = simple_loader
        
    def setup(self, stage=None):
        self.train_dataset = MyIterableMixedDataset(
            pos_readers = [(log['reader_path'], log['train_portion']) for log in self.config['pos'] if log['train_portion'] > 0],
            neg_readers = [(log['reader_path'], log['train_portion']) for log in self.config['neg'] if log['train_portion'] > 0],
            window = self.window,
            split='train',
        )
        self.valid_dataset = MyIterableMixedDataset(
            pos_readers = [(log['reader_path'], log['train_portion']) for log in self.config['pos'] if log['train_portion'] < 1],
            neg_readers = [(log['reader_path'], log['train_portion']) for log in self.config['neg'] if log['train_portion'] < 1],
            window = self.window,
            split='valid',
            limit=self.valid_limit,
        )
        
        assert self.train_dataset.vocab_map == self.vocab_map
        assert self.valid_dataset.vocab_map == self.vocab_map
        
        print('vocab ok')
        
            
    def train_dataloader(self):
        #No worker init function -> all workers can load all batches of preprocessed data (even the same)
        #more workers = more RAM used for caching batches
        #TODO dont refresh batches internally, but split them among workers permanently?
        if(self.simple_loader):
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True)
        
        return train_loader
    
    def val_dataloader(self):
        val_loader =  DataLoader(self.valid_dataset, batch_size=self.batch_size)
        return val_loader
    
class MyIterableMixedDataset(IterableDataset):
    def __init__(self, pos_readers, neg_readers, split, window, limit=None):
        pos_dsets = []
        for reader_path, train_portion in pos_readers:
            pos_dsets.append(MyIterableDataset(reader_path, window=window, split=split, train_portion=train_portion, replace_A_with_mod_A=True))
        neg_dsets = []
        for reader_path, train_portion in neg_readers:
            neg_dsets.append(MyIterableDataset(reader_path, window=window, split=split, train_portion=train_portion, replace_A_with_mod_A=False))
        control_vocab = pos_dsets[0].vocab_map
        for dset in pos_dsets+neg_dsets:
            assert dset.vocab_map == control_vocab
            control_vocab = dset.vocab_map
        
        self.vocab_map = control_vocab
        self.limit = limit
        self.pos_dsets = pos_dsets
        self.neg_dsets = neg_dsets
    
    def get_stream(self):
        pos_iterators = [iter(pos_dset) for pos_dset in self.pos_dsets]
        neg_iterators = [iter(neg_dset) for neg_dset in self.neg_dsets]
        
        pos_gen = uniform_gen(pos_iterators)
        neg_gen = uniform_gen(neg_iterators)
        
        mix_gen = alternating_gen([pos_gen, neg_gen])
        if(self.limit):
            for _ in range(self.limit):
                yield(next(mix_gen))
        else:
            while True:
                yield(next(mix_gen))
    
    def __iter__(self):
        return self.get_stream()
    

class MyIterableDataset(IterableDataset):
    def __init__(self, hdf5_reader_path, window, split, train_portion, replace_A_with_mod_A=False):
        self.reader = HDF5Reader(hdf5_reader_path)
        self.exp = Path(hdf5_reader_path).stem
        self.read_ids = self.reader.get_read_ids()
        self.window = window
        self.split = split
        self.train_portion=train_portion
        nums = self.reader.get_alphabet_information().collapse_labels
        letters = list(self.reader.get_alphabet_information().collapse_alphabet)
        self.vocab_map = {x:y for (x,y) in zip(letters,nums)}
        self.vocab_map_reversed = {v:k for k,v in self.vocab_map.items()}
        self.available_batch_names = self.reader.batch_names
        self.replace_A_with_mod_A = replace_A_with_mod_A
        
    def get_random_sample(self):
        mapping = random.choice(self.batch_mappings)
        signal = self.process_signal_mapping(mapping)
        
        #TODO add skip parameter instead of 0
        last_start_index = len(signal)-self.window
        #Using torch rand becasue of multiple workers
        start = torch.randint(0, last_start_index+1, (1,))
        end = start+self.window
        
        window_positions = (start,end)
        ref_beg, ref_end = np.searchsorted(mapping.Ref_to_signal, window_positions)
        bases_count  = ref_end-ref_beg
        
        window_ref = mapping.Reference[ref_beg:ref_end]
        
        #TODO how rodan limits data? base_len > smth etc...
        if(bases_count > 0):
            avg_spb = self.window/bases_count
            spb_ok = (avg_spb > 20) and (avg_spb < 70)
        else:
            spb_ok = False
            
        is_ok = (len(window_ref) == bases_count) and (bases_count > 0) and (spb_ok)
        event = signal[start:end]
        sequence = [self.vocab_map_reversed[index] for index in window_ref]
        
        return event, sequence, is_ok
        
    def process_signal_mapping(self, mapping):
        #Taken from RODAN code, replace with native mapping function
        signal = (mapping.Dacs + mapping.offset) * mapping.range / mapping.digitisation

        med = np.median(signal)
        mad = 1.4826 * np.median(abs(signal-med))
        signal = (signal - mapping.shift_frompA) / mapping.scale_frompA
        return signal
    
    def load_new_batch(self):
        new_batch_name = random.choice(self.available_batch_names)
        reads_batch = self.reader._load_reads_batch(new_batch_name)
        
        reads_in_batch = len(list(reads_batch.keys()))
        split_index = int(reads_in_batch*self.train_portion)
        if(self.split == 'train'):
            self.batch_mappings = list(reads_batch.values())[:split_index]
        if(self.split == 'valid'):
            self.batch_mappings = list(reads_batch.values())[split_index:]
        
        reads_batch_len = len(self.batch_mappings)
        
        #Sampling randomly for each element in batch before loading a new one
        self.batch_remaining_samples = reads_batch_len
        # print('Loading new batch', new_batch_name, 'size', reads_batch_len)
        
    
    def get_stream(self):
        # self.failures_before_yield = []
        # last_failures_before_yield = 0
        while True:
            # if(len(self.failures_before_yield)%100 == 0):
                # print(self.failures_before_yield)
            if(self.batch_remaining_samples <=0):
                self.load_new_batch()
            x,y, is_ok = self.get_random_sample()
            if(not is_ok):
                # last_failures_before_yield+=1
                continue
            if(self.replace_A_with_mod_A):
                y = ['X' if base=='A' else base for base in y]
            # self.failures_before_yield.append(last_failures_before_yield)
            # last_failures_before_yield = 0
            if(len(self.available_batch_names)>1):
                self.batch_remaining_samples -=1
            yield (np.array(x, dtype=np.float32),''.join(y), self.exp)

    def __iter__(self):
        self.load_new_batch()
        return self.get_stream()
        

