# RNAkinet - publication
RNAkinet is a project dedicated to detecting 5EU-modified reads from the raw nanopore sequencing signal. This repository contains the code for tha analysis of the RNAkinet publication. For the actual RNAkinet code refer to https://github.com/maragkakislab/rnakinet


## Custom training
1. Activate a conda environment with snakemake installed (you can use the `snakemake.yaml` file to create it)
2. Navigate to the `rnakinet/workflow` folder
3. Change the paths and parameters in `config/training_setup_custom.py` to reflect your data and requirements (provide genome fasta file, paths to fast5s etc...)
4. Open the `Snakemake` file and make sure the experiment name is the same one you specified in the `config/training_setup_custom.py` file
5. While in the workflow folder, run `snakemake --cores 32 --use-conda -np` to get a plan for execution. Once ready remove the `-np` flag to run training
6. Once training is finished, the model checkpoint will be available in the `checkpoints_pl` folder
7. You can use the checkpoint to run the `scripts/inference.py` to use it for prediction on other fast5 files