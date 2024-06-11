from pathlib import Path

class ExperimentData:
    def __init__(self, 
                 name=None, 
                 path=None, 
                 kit=None, 
                 flowcell=None, 
                 basecalls_path=None,
                 genome=None,
                 transcriptome=None,
                 train_chrs=None,
                 test_chrs=None,
                 valid_chrs=None,
                 halflives_name_to_file={},
                 time=None,
                 gene_transcript_table=None,
                ):
        self.name = name
        self.path = path
        self.kit = kit
        self.flowcell = flowcell
        self.basecalls_path = basecalls_path
        self.genome = genome
        self.transcriptome = transcriptome
        self.train_chrs = train_chrs
        self.test_chrs = test_chrs
        self.valid_chrs = valid_chrs
        self.halflives_name_to_file = halflives_name_to_file
        self.gene_transcript_table = gene_transcript_table
        # Time the cells have been exposed to 5EU, used for decay calculation
        self.time = time 
        
        if(not self.basecalls_path):
            #Default basecalls path
            self.basecalls_path = f"outputs/basecalling/{self.name}/guppy/reads.fastq.gz"
            
        self.splits_path = Path('outputs/splits')/self.name
        
    def get_name(self):
        return self.name
    def get_path(self):
        return self.path
    def get_kit(self):
        if(not self.kit):
            raise Exception(f'Kit not defined for {self.name}')
        return self.kit
    def get_flowcell(self):
        if(not self.kit):
            raise Exception(f'Flowcell not defined for {self.name}')
        return self.flowcell
    def get_basecalls(self):
        return self.basecalls_path
    def get_genome(self):
        if(not self.genome):
            raise Exception(f'Genome not defined for {self.name}')
        return self.genome
    def get_transcriptome(self):
        if(not self.transcriptome):
            raise Exception(f'Transcriptome not defined for {self.name}')
        return self.transcriptome
    def get_train_chrs(self):
        if(self.train_chrs is None):
            raise Exception(f'Train chromosomes not defined for {self.name}')
        return self.train_chrs
    def get_test_chrs(self):
        if(self.test_chrs is None):
            raise Exception(f'Test chromosomes not defined for {self.name}')
        return self.test_chrs
    def get_valid_chrs(self):
        if(self.valid_chrs is None):
            raise Exception(f'Validation chromosomes not defined for {self.name}')
        return self.valid_chrs
    def get_all_fast5_files(self):
        return list(self.get_path().rglob('*.fast5'))
    def get_train_fast5_files(self):
        return list((self.splits_path/'train').rglob('*.fast5'))
    def get_randomsplit_train_fast5_files(self):
        return list((self.splits_path/'randtrain').rglob('*fast5'))
    def get_test_fast5_files(self):
        return list((self.splits_path/'test').rglob('*.fast5'))
    def get_valid_fast5_files(self):
        return list((self.splits_path/'validation').rglob('*.fast5'))
    def get_split_fast5_files(self, split):
        getters = {
            'all':self.get_all_fast5_files,
            'train':self.get_train_fast5_files,
            'test':self.get_test_fast5_files,
            'validation':self.get_valid_fast5_files,
            'randtrain':self.get_randomsplit_train_fast5_files,
        }
        return getters[split]()
    def get_halflives_name_to_file(self):
        return self.halflives_name_to_file
    def get_time(self):
        if(self.time is None):
            raise Exception(f'Time is not defined for {self.name}')
        return self.time
    def get_gene_transcript_table(self):
        if(self.gene_transcript_table is None):
            raise Exception(f'Gene-transcript table not defined for {self.name}')
        return self.gene_transcript_table