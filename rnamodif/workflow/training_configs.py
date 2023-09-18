api_key = "TEVQbgxxvilM1WdTyqZLJ57ac"
default_train_positives = [
    'nia_2022_pos'
]
all_train_negatives = [
    'nia_2022_neg',
    'nia_2020_neg',
    'nia_neuron_hek',
    'nia_neuron_ctrl',
    'nia_neuron_tdp',
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0'
]
intermediate_train_negatives = [
    'nia_2022_neg',
    'nia_2020_neg',
    'nia_neuron_hek',
    'nia_neuron_ctrl',
    'nia_neuron_tdp',
]
basic_train_negatives = [
    'nia_2022_neg',
]
non2020_train_negatives = [
    'nia_2022_neg',
    'nia_neuron_hek',
    'nia_neuron_ctrl',
    'nia_neuron_tdp',
    'm6A_0',
    '2-OmeATP_0',
    'ac4C_0',
    'm5C_0',
    'remdesivir_0',
    's4U_0'
]
training_configs = {
    #TODO dont use name_to_files strings, unify paths from snakemake config and use here
    # '2022_allneg_w05': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':0.5,
    #     'wd':0.01,
    #     'arch':'custom_cnn',
    #     'arch_hyperparams':{
    #         'pooling':'max',
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':50, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
        # 'log_to_file':False,
    
    # },
    # '2022_rodan_allneg': {
    #     'training_positives_exps':default_train_positives,
    #     'training_negatives_exps':all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':0.5,
    #     'wd':0.01,
    #     'arch':'rodan',
    #     'arch_hyperparams':{
    #         'wd':0.01,
    #         'len_limit':400000,
    #         'weighted_loss':False,
    #         'frozen_layers':0,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':50, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
        # 'log_to_file':False,
    
    # },
    # '2022_cnn_rnn': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #     },
    #     'grad_acc':32,
    #     'early_stopping_patience':25, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_max': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_max',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #     },
    #     'grad_acc':32,
    #     'early_stopping_patience':25, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_att': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_att',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #     },
    #     'grad_acc':32,
    #     'early_stopping_patience':25, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    '2022_cnn_max_d3': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':3,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_d5': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_d7': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':7,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    
    '2022_cnn_max_hid5': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':10,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_intermediate_negs': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': intermediate_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_basic_negs': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': basic_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_intermediate_negs_uniform': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': intermediate_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'uniform',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_basic_negs_uniform': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': basic_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'uniform',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_d5_uniform': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'uniform',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100_dil2': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
            'dilation':2,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100_dil3': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
            'dilation':3,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100_min10000': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':10000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100_skip0': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':0,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid100_skip10000': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':10000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':100,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid200_skip10000': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':10000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':200,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid200_skip0': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':0,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':200,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    # 'TEST': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':100,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_max',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':10,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':50, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'yes',
    #     'log_to_file':False,
    # },
#     '2022_hybrid_nomaxpool_max_hid30': {
#         'training_positives_exps': default_train_positives,
#         'training_negatives_exps': all_train_negatives,
#         'min_len':5000,
#         'max_len':400000,
#         'skip':5000,
#         'workers':64,
#         'sampler':'ratio',
#         'lr':1e-3,
#         'warmup_steps':1000,
#         'pos_weight':1.0,
#         'wd':0.01,
#         'arch':'hybrid',
#         'arch_hyperparams':{
#             'cnn_depth':5,
#             'kernel_size':5,
#             'mlp_hidden_size':30,
#             'initial_channels':8,
            
#             'subrodan_depth':4,
#             #dependent on subrodan_depth
#             'subrodan_channels':320,
#         },
#         'grad_acc':64,
#         'early_stopping_patience':50, 
#         'comet_api_key':api_key,
#         'comet_project_name':'RNAModif',
#         'logging_step':500, 
#         'enable_progress_bar':'no',
#         'log_to_file':True,
#     },
    '2022_hybrid_max_dropout_hid10': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'hybrid',
        'arch_hyperparams':{
            'cnn_depth':3,
            'kernel_size':3,
            'mlp_hidden_size':10,
            'initial_channels':8,
            'subrodan_depth':4,
            #dependent on subrodan_depth
            'subrodan_channels':320,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
#     '2022_hybrid_max_hid30_non2020data': {
#         'training_positives_exps': default_train_positives,
#         'training_negatives_exps': non2020_train_negatives,
#         'min_len':5000,
#         'max_len':400000,
#         'skip':5000,
#         'workers':64,
#         'sampler':'ratio',
#         'lr':1e-3,
#         'warmup_steps':1000,
#         'pos_weight':1.0,
#         'wd':0.01,
#         'arch':'hybrid',
#         'arch_hyperparams':{
#             'cnn_depth':4,
#             'kernel_size':3,
#             'mlp_hidden_size':30,
#             'initial_channels':8,
            
#             'subrodan_depth':4,
#             #dependent on subrodan_depth
#             'subrodan_channels':320,
#         },
#         'grad_acc':64,
#         'early_stopping_patience':50, 
#         'comet_api_key':api_key,
#         'comet_project_name':'RNAModif',
#         'logging_step':500, 
#         'enable_progress_bar':'no',
#         'log_to_file':True,
#     },
    '2022_hybrid_max_dropout_hid10_min15k': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':15000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'hybrid',
        'arch_hyperparams':{
            'cnn_depth':3,
            'kernel_size':3,
            'mlp_hidden_size':10,
            'initial_channels':8,
            'subrodan_depth':4,
            #dependent on subrodan_depth
            'subrodan_channels':320,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    '2022_cnn_max_hid30_min15k': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':15000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':1.0,
        'wd':0.01,
        'arch':'cnn_max',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':30,
        },
        'grad_acc':64,
        'early_stopping_patience':50, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },
    # 'rodanlike_max_hid30': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'RODANlike',
    #     'arch_hyperparams':{
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':50, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    
    #TODO rnn min15 min5 + 
    # '2022_cnn_rnn_hid30': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_rnn_hid30_min15k': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':15000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_max_hid30_min15k_longtrain': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':15000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_max',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_rnn_hid30_run2': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_rnn_hid30_basicneg': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': basic_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    # '2022_cnn_rnn_hid30_allneg_run2': {
    #     'training_positives_exps': default_train_positives,
    #     'training_negatives_exps': all_train_negatives,
    #     'min_len':5000,
    #     'max_len':400000,
    #     'skip':5000,
    #     'workers':64,
    #     'sampler':'ratio',
    #     'lr':1e-3,
    #     'warmup_steps':1000,
    #     'pos_weight':1.0,
    #     'wd':0.01,
    #     'arch':'cnn_rnn',
    #     'arch_hyperparams':{
    #         'cnn_depth':5,
    #         'mlp_hidden_size':30,
    #     },
    #     'grad_acc':64,
    #     'early_stopping_patience':150, 
    #     'comet_api_key':api_key,
    #     'comet_project_name':'RNAModif',
    #     'logging_step':500, 
    #     'enable_progress_bar':'no',
    #     'log_to_file':True,
    # },
    '2022_cnn_rnn_hid10_allneg_w05': {
        'training_positives_exps': default_train_positives,
        'training_negatives_exps': all_train_negatives,
        'min_len':5000,
        'max_len':400000,
        'skip':5000,
        'workers':64,
        'sampler':'ratio',
        'lr':1e-3,
        'warmup_steps':1000,
        'pos_weight':0.5,
        'wd':0.01,
        'arch':'cnn_rnn',
        'arch_hyperparams':{
            'cnn_depth':5,
            'mlp_hidden_size':10,
        },
        'grad_acc':64,
        'early_stopping_patience':150, 
        'comet_api_key':api_key,
        'comet_project_name':'RNAModif',
        'logging_step':500, 
        'enable_progress_bar':'no',
        'log_to_file':True,
    },

}