from pathlib import Path
from config.config_helpers import ExperimentData
    
# References downloaded from ENSEMBL
# transcript-gene.tab downloaded and renamed from 
# http://useast.ensembl.org/biomart/martview/989e4fff050168c3154e5398a6f27dde

mouse_genome = 'references/Mus_musculus.GRCm39.dna_sm.primary_assembly.fa'
mouse_transcriptome = 'references/Mus_musculus.GRCm39.cdna.all.fa'
mouse_gene_transcript_table = 'references/transcript-gene-mouse.tab'

human_genome = 'references/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'
human_transcriptome = 'references/Homo_sapiens.GRCh38.cdna.all.fa'
human_gene_transcript_table = 'references/transcript-gene-human.tab'
    
experiments_data = {}
   
#TODO remove & rename
inhouse_exps = [
    '20220520_hsa_dRNA_HeLa_DMSO_1', 
    '20220520_hsa_dRNA_HeLa_5EU_200_1',
    
    '20220303_hsa_dRNA_HeLa_DMSO_polyA_REL5_2',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
]

for exp_name in inhouse_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/store/seq/ont/experiments/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[20],
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

nanoid_exps = [
    '20180327_1102_K562_5EU_0_unlabeled_run',
    '20180403_1102_K562_5EU_0_unlabeled_II_run',
    '20180403_1208_K562_5EU_0_unlabeled_III_run',
    '20180514_1054_K562_5EU_1440_labeled_run',
    '20180514_1541_K562_5EU_1440_labeled_II_run',
    '20180516_1108_K562_5EU_1440_labeled_III_run',
    
    '20180226_1208_K562_5EU_60_labeled_run',
    '20180227_1206_K562_5EU_60_labeled_II_run',
    '20180228_1655_K562_5EU_60_labeled_III_run',
    '20181206_1038_K562_5EU_60_labeled_IV_run',
    '20190719_1232_K562_5EU_60_labeled_V_run',
    '20190719_1430_K562_5EU_60_labeled_VI_run',
    
    '20180628_1020_K562_5EU_60_labeled_heat_run',
    '20180731_1020_K562_5EU_60_labeled_heat_II_run',
    '20180802_1111_K562_5EU_60_labeled_heat_III_run',
    '20190725_0809_K562_5EU_60_labeled_heat_IV_run',
    '20190725_0812_K562_5EU_60_labeled_heat_V_run',
]
for exp_name in nanoid_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/nanoid/{exp_name}',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[],
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    
two_hour_5eu_exps = [    
    # TODO resolve joined ALL_NoArs60 experiment for tani halflives plotting
    '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
    '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
    '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
]

for exp_name in two_hour_5eu_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        halflives_name_to_file={'DRB':'halflives_data/experiments/hl_drb_renamed.tsv'}, #tani halflives
        time=2.0,
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data


neuron_exps = [    
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
]
for exp_name in neuron_exps:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=human_genome,
        transcriptome=human_transcriptome,
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[20],
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    
external_mouse = [
    '20211203_mmu_dRNA_3T3_mion_1',
    '20211203_mmu_dRNA_3T3_PION_1',
]
for exp_name in external_mouse:
    exp_data = ExperimentData(
        name=exp_name,
        path=f'local_store/arsenite/raw/{exp_name}',
        genome=mouse_genome,
        transcriptome=mouse_transcriptome,
        gene_transcript_table=mouse_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

inhouse_mouse = [
    '20230706_mmu_dRNA_3T3_5EU_400_1',
    '20230816_mmu_dRNA_3T3_5EU_400_2',
]

for exp_name in inhouse_mouse:
    root_dir=Path(f'local_store/arsenite/raw/{exp_name}')
    fast5_pass_dirs = [x for x in root_dir.glob("**/fast5_pass") if x.is_dir()]
    assert len(fast5_pass_dirs) == 1, len(fast5_pass_dirs)
    exp_path = fast5_pass_dirs[0]
    exp_data = ExperimentData(
        name=exp_name,
        path=exp_path,
        kit='SQK-RNA002',
        flowcell='FLO-MIN106',
        genome=mouse_genome,
        transcriptome=mouse_transcriptome,
        halflives_name_to_file={
            'PION':'halflives_data/experiments/mmu_dRNA_3T3_PION_1/features_v1.tsv',
            'MION':'halflives_data/experiments/mmu_dRNA_3T3_mion_1/features_v1.tsv',
        },
        time=2.0,
        gene_transcript_table=mouse_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data
    
    
#TODO remove _TEST splits (now needed for plotting)
test_splits = [
    
    '20180327_1102_K562_5EU_0_unlabeled_run_TEST',
    '20180514_1054_K562_5EU_1440_labeled_run_TEST',
    
    '20180403_1102_K562_5EU_0_unlabeled_II_run_TEST',
    '20180514_1541_K562_5EU_1440_labeled_II_run_TEST',
    
    '20180403_1208_K562_5EU_0_unlabeled_III_run_TEST',
    '20180516_1108_K562_5EU_1440_labeled_III_run_TEST',
    
    '20220520_hsa_dRNA_HeLa_DMSO_1_TEST',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TEST',
    

]
for exp_name in test_splits:
    og_exp_name = exp_name[:-5]
    exp_data = ExperimentData(
        name=exp_name,
        path=f'outputs/splits/{og_exp_name}/test',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        # kit=,
        # flowcell=,
        genome=human_genome,
        transcriptome=human_transcriptome,
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

#TODO remove _TRAIN splits (now needed for plotting)
train_splits = [
    '20180327_1102_K562_5EU_0_unlabeled_run_TRAIN',
    '20180514_1054_K562_5EU_1440_labeled_run_TRAIN',
    
    '20180403_1102_K562_5EU_0_unlabeled_II_run_TRAIN', 
    '20180514_1541_K562_5EU_1440_labeled_II_run_TRAIN', 
    
    '20180403_1208_K562_5EU_0_unlabeled_III_run_TRAIN', 
    '20180516_1108_K562_5EU_1440_labeled_III_run_TRAIN', 
    
    '20220520_hsa_dRNA_HeLa_DMSO_1_TRAIN',
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TRAIN',
]
for exp_name in train_splits:
    og_exp_name = exp_name[:-6]
    exp_data = ExperimentData(
        name=exp_name,
        path=f'outputs/splits/{og_exp_name}/train',
        kit='SQK-RNA001',
        flowcell='FLO-MIN106',
        # kit=,
        # flowcell=,
        genome=human_genome,
        transcriptome=human_transcriptome,
        gene_transcript_table=human_gene_transcript_table,
    )
    experiments_data[exp_name] = exp_data

# Used for concatenating all replicates
experiments_data['ALL_NoArs60'] = ExperimentData(
    name='ALL_NoArs60',
    genome=human_genome,
    transcriptome=human_transcriptome,
    gene_transcript_table=human_gene_transcript_table,
    time=2.0,
    halflives_name_to_file={'DRB':'halflives_data/experiments/hl_drb_renamed.tsv'}, #tani halflives
)

r10_exps_human = [
    "20240410_hsa_dRNA_HeLa_GFP_NoARS_1",
    "20240503_hsa_dRNA_HeLa_GFP_24h_NoARS_1",
    "20240510_hsa_dRNA_HeLa_5EU_R10_1",
    '20240214_hsa_dRNA_iN3_TDP43_WT_1',
    '20240510_hsa_dRNA_HeLa_5EU_R10_1',
]
#TODO basecalls - should be all? Or just pass files (all_reads VS reads)
for exp_name in r10_exps_human:
    root_dir=Path(f'local_store/arsenite/raw/{exp_name}/runs/no_sample')
    fast5_pass_dirs = [x for x in root_dir.glob("**/fast5_pass") if x.is_dir()]
    assert len(fast5_pass_dirs) == 1, len(fast5_pass_dirs)
    exp_data = ExperimentData(
        name=exp_name,
        path=fast5_pass_dirs[0],
        kit='SQK-RNA004',
        flowcell='FLO-PRO004RA',
        genome=human_genome, 
        transcriptome=human_transcriptome, 
        train_chrs=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','MT'],
        test_chrs=[1],
        valid_chrs=[20],
        gene_transcript_table=human_gene_transcript_table, 
        basecalls_path = f'outputs/basecalling/{exp_name}/dorado/all_reads.fastq.gz', #TODO generalize for all dorado-basecalled experiments
    )
    experiments_data[exp_name] = exp_data
    
    
r10_exps_mouse = [
    "20240502_mmu_dRNA_3T3_5EU_400_2_R10",
]
for exp_name in r10_exps_mouse:
    root_dir=Path(f'local_store/arsenite/raw/{exp_name}/runs/no_sample')
    fast5_pass_dirs = [x for x in root_dir.glob("**/fast5_pass") if x.is_dir()]
    assert len(fast5_pass_dirs) == 1, len(fast5_pass_dirs)
    exp_data = ExperimentData(
        name=exp_name,
        path=fast5_pass_dirs[0],
        kit='SQK-RNA004',
        flowcell='FLO-PRO004RA',
        genome=mouse_genome, 
        transcriptome=mouse_transcriptome, 
        gene_transcript_table=mouse_gene_transcript_table, 
        basecalls_path = f'outputs/basecalling/{exp_name}/dorado/all_reads.fastq.gz', #TODO generalize for all dorado-basecalled experiments
    )
    experiments_data[exp_name] = exp_data


default_positives = [
    '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'
]
default_negatives = [
    '20220520_hsa_dRNA_HeLa_DMSO_1',
    '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
    '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
    '20201022_hsa_dRNA_Neuron_TDP_5P_1',
]

api_key = "y4EULBjxNd83yrzrwaLuxHtcr"

training_configs  = {
    'rnakinet': {
        'training_positives_exps': default_positives,
        'training_negatives_exps': default_negatives,
        'validation_positives_exps': default_positives,
        'validation_negatives_exps': default_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':100,
        'wd':0.01,
        'arch':'rnakinet',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':10,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAkinet',
        'logging_step':500,
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    'deploy_test': {
        'training_positives_exps': default_positives,
        'training_negatives_exps': default_negatives,
        'validation_positives_exps': default_positives,
        'validation_negatives_exps': default_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':32,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':100,
        'wd':0.01,
        'arch':'rnakinet',
        'arch_hyperparams':{},
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAkinet',
        'logging_step':500,
        'enable_progress_bar':'no',
        'log_to_file':True,
        'valid_read_limit':5000,
    },
}