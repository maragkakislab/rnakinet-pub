import random
import pytorch_lightning as pl
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
from torch.utils.data import IterableDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnakinet.data_utils.generators import uniform_gen, ratio_gen
from rnakinet.data_utils.read_utils import process_read
from rnakinet.data_utils.workers import worker_init_fn

class TrainingDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            train_pos_lists,
            train_neg_lists,
            valid_pos_lists,
            valid_neg_lists,
            batch_size,
            valid_read_limit,
            shuffle_valid,
            workers,
            max_len,
            skip,
            multiexp_generator_type,
            min_len,
    ):

        super().__init__()
        self.train_pos_lists = train_pos_lists
        self.train_neg_lists = train_neg_lists
        
        self.valid_pos_lists = valid_pos_lists
        self.valid_neg_lists = valid_neg_lists

        check_leakage(
            train_pos_lists = self.train_pos_lists, 
            train_neg_lists = self.train_neg_lists, 
            valid_pos_lists = self.valid_pos_lists, 
            valid_neg_lists = self.valid_neg_lists, 
        )
        
        self.batch_size = batch_size
        self.valid_read_limit = valid_read_limit
        self.shuffle_valid = shuffle_valid
        self.workers = workers

        self.train_dataset = None
        self.valid_dataset = None
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len
        
        self.multiexp_generator_type = multiexp_generator_type

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None):
            self.train_dataset = UnlimitedReadsTrainingDataset(
                pos_lists=self.train_pos_lists,
                neg_lists=self.train_neg_lists,
                max_len = self.max_len,
                skip=self.skip,
                min_len=self.min_len,
                multiexp_generator_type=self.multiexp_generator_type,
            )
            self.valid_dataset = UnlimitedReadsValidDataset(
                pos_lists=self.valid_pos_lists,
                neg_lists=self.valid_neg_lists,
                valid_read_limit=self.valid_read_limit,
                skip=self.skip,
                max_len=self.max_len,
                min_len=self.min_len,
                multiexp_generator_type='uniform',
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self):
        return  DataLoader(self.valid_dataset, batch_size=self.batch_size)


    
class UnlimitedReadsTrainingDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """
    def __init__(self, pos_lists, neg_lists, max_len, skip, min_len, multiexp_generator_type):
        self.positive_lists = pos_lists
        self.negative_lists = neg_lists
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type

    def process_files(self, files, label, exp):
        while True:
            fast5 = random.choice(files)
            # try:
            with get_fast5_file(fast5, mode='r') as f5:
                reads = list(f5.get_reads())
                random.shuffle(reads)
                for read in reads:
                    x = process_read(read, skip=self.skip)
                    y = np.array(label)
                    # Skip if the read is too short
                    if (len(x) > self.max_len or len(x) < self.min_len):
                        continue
                    yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32), exp

    def get_stream(self):
        pos_gens = []
        pos_sizes = []
        for pos_files in self.positive_lists:
            assert len(pos_files) > 0, pos_files
            pos_gens.append(self.process_files(files=pos_files, label=1, exp='pos'))
            pos_sizes.append(len(pos_files))
        neg_gens = []
        neg_sizes = []
        for neg_files in self.negative_lists:
            assert len(neg_files) > 0, neg_files
            neg_gens.append(self.process_files(files=neg_files, label=0, exp='neg'))
            neg_sizes.append(len(neg_files))
            
        if(self.multiexp_generator_type  == 'uniform'):
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)
        
        if(self.multiexp_generator_type == 'ratio'):
            pos_ratios = np.array(pos_sizes)/np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes)/np.sum(neg_sizes)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        
        while True:
            yield next(gen)

    def __iter__(self):
        return self.get_stream()

class UnlimitedReadsInferenceDataset(IterableDataset):
    """
    Iterable Dataset that contains all reads
    """

    def __init__(self, files, max_len, skip, min_len):
        self.files = files
        self.max_len = max_len
        self.skip = skip
        self.min_len = min_len

    def process_files_fully(self, files):
        for fast5 in files:
            try:
                with get_fast5_file(fast5, mode='r') as f5:
                    for i, read in enumerate(f5.get_reads()):
                        x = process_read(read, skip=self.skip)
                        start = 0
                        stop = len(x)
                        if(len(x) > self.max_len or len(x) < self.min_len):
                            continue
                        identifier = {
                            'file': str(fast5),
                            'readid': read.read_id,
                            'read_index_in_file': 0,
                            'start': start,
                            'stop': stop,
                        }
                        yield x.reshape(-1, 1).swapaxes(0, 1), identifier
            except OSError as error:
                print(error)
                continue

    def __iter__(self):
        return self.process_files_fully(self.files)
    
    
class UnlimitedReadsValidDataset(Dataset):
    """
    Mapped Dataset that contains validation reads
    """
    def __init__(self, pos_lists, neg_lists, max_len, skip, min_len, multiexp_generator_type, valid_read_limit):
        self.positive_lists = pos_lists
        self.negative_lists = neg_lists
        self.max_len = max_len
        self.min_len = min_len
        self.skip = skip
        self.multiexp_generator_type = multiexp_generator_type
        self.valid_read_limit = valid_read_limit
        
        pos_gens = []
        for lists in self.positive_lists:
            for files in lists:
                pos_gens.append(self.process_files(
                    files, label=1))
        neg_gens = []
        for lists in self.negative_lists:
            for files in lists:
                neg_gens.append(self.process_files(
                    files, label=0))
        
        print('Generating valid dataset')
        self.items = self.generate_data()

    def process_files(self, files, label):
        while True:
            fast5 = random.choice(files)
            with get_fast5_file(fast5, mode='r') as f5:
                reads = list(f5.get_reads())
                for read in reads:
                    x = process_read(read, skip=self.skip)
                    y = np.array(label)
                    # Skip if the read is too short
                    if (len(x) > self.max_len or len(x) < self.min_len):
                        continue
                    yield x.reshape(-1, 1).swapaxes(0, 1), np.array([y], dtype=np.float32)

    def generate_data(self):
        pos_gens = []
        pos_sizes = []
        for pos_files in self.positive_lists:
            assert len(pos_files) > 0, pos_files
            pos_gens.append(self.process_files(files=pos_files, label=1))
            pos_sizes.append(len(pos_files))
        neg_gens = []
        neg_sizes = []
        for neg_files in self.negative_lists:
            assert len(neg_files) > 0, neg_files
            neg_gens.append(self.process_files(files=neg_files, label=0))
            neg_sizes.append(len(neg_files))
            
        if(self.multiexp_generator_type  == 'uniform'):
            global_pos_gen = uniform_gen(pos_gens)
            global_neg_gen = uniform_gen(neg_gens)
        
        if(self.multiexp_generator_type == 'ratio'):
            pos_ratios = np.array(pos_sizes)/np.sum(pos_sizes)
            neg_ratios = np.array(neg_sizes)/np.sum(neg_sizes)
            global_pos_gen = ratio_gen(pos_gens, pos_ratios)
            global_neg_gen = ratio_gen(neg_gens, neg_ratios)
        
        gen = uniform_gen([global_pos_gen, global_neg_gen])
        
        items = []
        for _ in range(self.valid_read_limit):
            x, y = next(gen)
            items.append((x, y))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
    
# CHECK DATA LEAKAGE TEST
def check_leakage(train_pos_lists, train_neg_lists, valid_pos_lists, valid_neg_lists):
    train_sets = []
    for file_array in train_neg_lists+train_pos_lists:
        train_sets.append(set(file_array))

    assert(len(set.intersection(*train_sets))==0)
    
    valid_sets = []
    for file_array in valid_pos_lists+valid_neg_lists:
        valid_sets.append(set(file_array))

    assert(len(set.intersection(*valid_sets))==0)
        
    train_set = set.union(*train_sets)
    valid_set = set.union(*valid_sets)

    assert(len(set.intersection(train_set, valid_set))==0)