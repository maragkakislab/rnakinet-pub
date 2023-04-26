from rnamodif.architectures.rodan_pretrained_modcaller import RodanPretrainedModcaller
from rnamodif.data_utils.dataloading2 import nanopore_datamodule
from rnamodif.data_utils.split_methods import get_kfold_splits, get_fullvalid_split, get_valid_portions
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

model = RodanPretrainedModcaller(lr=1e-4, use_Ns_for_loss=False, weighted_loss=False, use_basecalling_loss=True)

dm = nanopore_datamodule(
    splits = get_kfold_splits(
        pos_exps=['m6a_novoa_1', 'm6a_novoa_2'],
        neg_exps=['UNM_novoa_1', 'UNM_novoa_2', 'UNM_novoa_short'], 
        total_k=5, 
        current_k=0
    )
    +get_fullvalid_split()(pos_files=['m6A_5_covid','m6A_10_covid','m6A_33_covid'], neg_files=['m6A_0_covid']),
    batch_size=32, 
    valid_limit=1000, 
    workers=4,
    window=4096,
    normalization='rodan',
)

experiment_name = 'm6a_novoa_base_unweighted_noN_basecalling'
checkpoint_callback = ModelCheckpoint(
    dirpath=f"/home/jovyan/RNAModif/rnamodif/checkpoints_pl/{experiment_name}", 
    save_top_k=2, 
    monitor="valid_loss", 
    save_last=True, 
    save_weights_only=False
)

logger = CometLogger(api_key="TEVQbgxxvilM1WdTyqZLJ57ac", project_name='RNAModif', experiment_name=experiment_name) 
trainer= pl.Trainer(
    max_steps = 1000000, logger=logger, accelerator='gpu',
    auto_lr_find=False, val_check_interval=500,  
    log_every_n_steps=500, benchmark=True, precision=16,
    callbacks=[checkpoint_callback],
    resume_from_checkpoint='/home/jovyan/RNAModif/rnamodif/checkpoints_pl/m6a_novoa_base_unweighted_noN_basecalling/last.ckpt')


trainer.fit(model, dm)