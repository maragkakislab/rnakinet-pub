class ModelInferenceInfo():
    def __init__(self, path, arch, batch_size, max_len, min_len, skip, threshold):
        self.path = path
        self.arch = arch
        self.batch_size = batch_size
        self.max_len = max_len
        self.min_len = min_len
        # How many raw signal values to skip at the beginning of reads (signals are noisy at the beginning)
        self.skip = skip
        self.threshold = threshold
    def get_path(self):
        return self.path
    def get_arch(self):
        return self.arch
    def get_batch_size(self):
        return self.batch_size
    def get_max_len(self):
        return self.max_len
    def get_min_len(self):
        return self.min_len
    def get_skip(self):
        return self.skip
    def get_threshold(self):
        return self.threshold

models_data = {}
model_name_to_args = {
    'rnakinet':{
        'path':'../checkpoints_pl/rnakinet/last-Copy5.ckpt',
        'threshold': 0.5,
        'arch': 'rnakinet',
    },
    'rnakinet_r10_lastonly':{
        'path':'../checkpoints_pl/rnakinet_r10_lastonly/best-step=46500-valid_loss=0.41.ckpt',
        'threshold':0.5,
        'arch':'rnakinet_lastonly',
    }
}

for model_name, model_args in model_name_to_args.items():
    models_data[model_name] = ModelInferenceInfo(
        path=model_args['path'],
        arch=model_args['arch'],
        batch_size=1,
        max_len=400000,
        min_len=5000,
        skip=5000,
        threshold=model_args['threshold'],
    )

#Evaluation data 
#TODO unify with training data??
exp_groups = {
    'all_nanoid_positives':[
        '20180514_1054_K562_5EU_1440_labeled_run',
        '20180514_1541_K562_5EU_1440_labeled_II_run',
        '20180516_1108_K562_5EU_1440_labeled_III_run',
    ],
    'all_nanoid_negatives':[
        '20180327_1102_K562_5EU_0_unlabeled_run',
        '20180403_1102_K562_5EU_0_unlabeled_II_run',
        '20180403_1208_K562_5EU_0_unlabeled_III_run',
    ],
    'all_2022_nia_positives':[
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2'
    ],
    'all_2022_nia_negatives':[
        '20220520_hsa_dRNA_HeLa_DMSO_1'
    ],
    'test_nanoid_positives':[
        '20180514_1054_K562_5EU_1440_labeled_run_TEST',
        '20180514_1541_K562_5EU_1440_labeled_II_run_TEST',
        '20180516_1108_K562_5EU_1440_labeled_III_run_TEST',
    ],
    'test_nanoid_negatives':[
        '20180327_1102_K562_5EU_0_unlabeled_run_TEST',
        '20180403_1102_K562_5EU_0_unlabeled_II_run_TEST',
        '20180403_1208_K562_5EU_0_unlabeled_III_run_TEST',
    ],
    'test_2022_nia_positives': [
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2_TEST'
    ],
    'test_2022_nia_negatives': [
        '20220520_hsa_dRNA_HeLa_DMSO_1_TEST'
    ],
    'nanoid_shock_controls':[
        '20180226_1208_K562_5EU_60_labeled_run',
        '20180227_1206_K562_5EU_60_labeled_II_run',
        '20180228_1655_K562_5EU_60_labeled_III_run',
        '20181206_1038_K562_5EU_60_labeled_IV_run',
        '20190719_1232_K562_5EU_60_labeled_V_run',
        '20190719_1430_K562_5EU_60_labeled_VI_run',
    ],
    'nanoid_shock_conditions':[
        '20180628_1020_K562_5EU_60_labeled_heat_run',
        '20180731_1020_K562_5EU_60_labeled_heat_II_run',
        '20180802_1111_K562_5EU_60_labeled_heat_III_run',
        '20190725_0809_K562_5EU_60_labeled_heat_IV_run',
        '20190725_0812_K562_5EU_60_labeled_heat_V_run',
    ],
    'stat_exps':[
        '20180514_1054_K562_5EU_1440_labeled_run',
        '20180514_1541_K562_5EU_1440_labeled_II_run',
        '20180516_1108_K562_5EU_1440_labeled_III_run',
        '20180327_1102_K562_5EU_0_unlabeled_run',
        '20180403_1102_K562_5EU_0_unlabeled_II_run',
        '20180403_1208_K562_5EU_0_unlabeled_III_run',
        '20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2',
        '20220520_hsa_dRNA_HeLa_DMSO_1',
    ],
    'hela_decay_exps':[
        '20201215_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_1',
        '20210202_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_2',
        '20210519_hsa_dRNA_HeLa_5EU_2hr_NoArs_0060m_5P_3',
    ],
    'mouse_decay_exps':[
        '20230706_mmu_dRNA_3T3_5EU_400_1',
        '20230816_mmu_dRNA_3T3_5EU_400_2',
    ],
    'neuron_exps':[
        '20201001_hsa_dRNA_Hek293T_NoArs_5P_1',
        '20201022_hsa_dRNA_Neuron_ctrl_5P_1',
        '20201022_hsa_dRNA_Neuron_TDP_5P_1',
    ],
    'r10_mouse_decay_exps':[
        '20240502_mmu_dRNA_3T3_5EU_400_2_R10',
    ],
    'test_r10_HeLa_positives':[
        '20240510_hsa_dRNA_HeLa_5EU_R10_1_TEST',
    ],
    'test_r10_HeLa_negatives':[
        "20240410_hsa_dRNA_HeLa_GFP_NoARS_1_TEST",
        "20240503_hsa_dRNA_HeLa_GFP_24h_NoARS_1_TEST",
        '20240214_hsa_dRNA_iN3_TDP43_WT_1_TEST',
    ],
    # 'test_2020_nia_positives':[
    #    '20201016_hsa_dRNASeq_HeLa_5EU_polyA_REL5_short_1_TEST',
    # ],
    # 'test_2020_nia_negatives':[
    #    '20201016_hsa_dRNASeq_HeLa_dmso_polyA_REL5_short_1_TEST',
    # ],
}

pos_neg_pairs = {
    'ALL_NANOID':{
        'positives':exp_groups['all_nanoid_positives'],
        'negatives':exp_groups['all_nanoid_negatives'],
    },
    'TEST_NANOID':{
        'positives':exp_groups['test_nanoid_positives'],
        'negatives':exp_groups['test_nanoid_negatives'],
    },
    'ALL_2022_NIA':{
        'positives':exp_groups['all_2022_nia_positives'],
        'negatives':exp_groups['all_2022_nia_negatives'],    
    },
    'TEST_2022_NIA':{
        'positives':exp_groups['test_2022_nia_positives'],
        'negatives':exp_groups['test_2022_nia_negatives'],
    },
    'TEST_2022_NIA_R10':{
        'positives':exp_groups['test_r10_HeLa_positives'],
        'negatives':exp_groups['test_r10_HeLa_negatives'],
    },
    # 'TEST_2020_NIA':{
    #     'positives':exp_groups['test_2020_nia_positives'],
    #     'negatives':exp_groups['test_2020_nia_negatives'],
    # },
}

condition_control_pairs = {
    'NANOID_shock': {
        'controls':exp_groups['nanoid_shock_controls'],
        'conditions':exp_groups['nanoid_shock_conditions'],
    },
}

comparison_groups = {
    'ALL':['ALL_2022_NIA','ALL_NANOID'],
    'TEST':['TEST_2022_NIA','TEST_NANOID'],
    'TEST_R10':['TEST_2022_NIA_R10'],
    # 'TEST':['TEST_2022_NIA','TEST_NANOID', 'TEST_2020_NIA'],
}

model_comparison_groups = {
    'ALL':['rnakinet'],
}

datastats_groups = {
    'nanoid':[item for group in ['all_nanoid_positives','all_nanoid_negatives'] for item in exp_groups[group]],
    'nanoid_shock':[item for group in ['nanoid_shock_controls','nanoid_shock_conditions'] for item in exp_groups[group]],
    'noars60':[item for group in ['hela_decay_exps'] for item in exp_groups[group]],
    '3t3':[item for group in ['mouse_decay_exps'] for item in exp_groups[group]],
    'neurons': [item for group in ['neuron_exps'] for item in exp_groups[group]],
    'training': [item for group in ['all_2022_nia_positives','all_2022_nia_negatives','neuron_exps'] for item in exp_groups[group]],
}

pooling=['max'] #TODO refactor away