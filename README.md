# RNAModif
Project to detect 5eu-modified reads from raw nanopore sequencing signal

# Instalation
```sh
git clone https://github.com/maragkakislab/RNAModif.git
conda create -n snakemake_env -c bioconda snakemake
conda activate snakemake_env
pip install -e .
```


# Usage
Open ```RNAModif/rnamodif/workflow/Snakemake``` file and add your data path to the ```name_to_fast5_path``` dictionary. Then run:

```sh
cd RNAModif/rnamodif/workflow
snakemake --cores 64 --use-conda
```

This generates prediction csv file in ```RNAModif/rnamodif/workflow/outputs/predictions/CUSTOM_allneg_maxpool/<your_experiment_name>/max_pooling.csv```

The file contains the following columns:

```read_id``` - the read identifier from the fast5 file

```5eu_mod_score``` - the 5eu modification score of the read (from 0 to 1)

```5eu_modified_prediction``` - the 5eu modification prediction of the read (True = 5eu modified, False = unmodified) based on the best model threshold


