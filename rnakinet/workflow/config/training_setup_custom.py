from pathlib import Path
from config.config_helpers import ExperimentData


#Set your references
human_genome = 'references/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'
experiments_data = {}

r10_exps_negatives = [
    "20240410_hsa_dRNA_HeLa__1",
]
r10_exps_positives = [
    "20240510_hsa_dRNA_HeLa_5EU_1",
]
for exp_name in r10_exps_negatives+r10_exps_positives:
    #Set paths to fast5 files for each experiment
    fast5_pass_dir=Path(f'datastore/{exp_name}/fast5_pass')
    
    exp_data = ExperimentData(
        name=exp_name,
        path=fast5_pass_dir,
        genome=human_genome, 
        kit='SQK-RNA004', #set your kit
        flowcell='FLO-PRO004RA', #set your flowcell
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','MT'], #chromosomes used for training
        test_chrs=[1], #chromosomes used for testing
        valid_chrs=[20], #chromosomes used for validation
        basecalls_path = f'outputs/basecalling/{exp_name}/dorado/all_reads.fastq.gz', #Change this path for your own file if you have your own basecalls and do not wish to re-basecall with dorado. Must be in fastq.gz format.
    )
    experiments_data[exp_name] = exp_data
    

training_configs  = {
    'test_run': {
        'training_positives_exps': r10_exps_positives,
        'training_negatives_exps': r10_exps_negatives,
        'validation_positives_exps': r10_exps_positives, #Same experiment, but will take reads from a different chromosome
        'validation_negatives_exps': r10_exps_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio', #sample from experiments based on their size, alternative 'uniform' for uniform sampling
        'lr':1e-3,
        'warmup_steps':100,
        'wd':0.01,
        'arch':'rnakinet',
        'arch_hyperparams':{},
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':'y4EULBjxNd83yrzrwaLuxHtcr', #OPTIONAL: Set your comet API key for plotting training metrics
        'comet_project_name':'RNAkinet',
        'logging_step':500,
        'enable_progress_bar':'yes', #set to yes if you want to log training progress bar
        'log_to_file':False, #set to False if you want to log to console, otherwise logged to checkpoints_pl folder
        'valid_read_limit':5000,
    },
}