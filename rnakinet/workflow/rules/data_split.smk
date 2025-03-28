rule split_readids_on_chromosomes:
    input:
        bam_path="outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam",
    output:
        "outputs/splits/{experiment_name}/train_readids.txt",
        "outputs/splits/{experiment_name}/test_readids.txt",
        "outputs/splits/{experiment_name}/validation_readids.txt",
    conda:
        "../envs/bam_splitting.yaml"
    params:
        train_chromosomes=lambda wildcards: experiments_data[wildcards.experiment_name].get_train_chrs(),
        test_chromosomes=lambda wildcards: experiments_data[wildcards.experiment_name].get_test_chrs(),
        validation_chromosomes=lambda wildcards: experiments_data[wildcards.experiment_name].get_valid_chrs(),
    shell:
        """
        python3 scripts/splitting.py \
            --bam_path {input.bam_path} \
            --output_path outputs/splits/{wildcards.experiment_name}/ \
            --train_chromosomes {params.train_chromosomes} \
            --test_chromosomes {params.test_chromosomes} \
            --validation_chromosomes {params.validation_chromosomes} \
        """
        
rule split_readids_random:
    input:
        bam_path="outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam",
    output:
        "outputs/splits/{experiment_name}/randtrain_readids.txt",
        "outputs/splits/{experiment_name}/randtest_readids.txt",
        "outputs/splits/{experiment_name}/randvalidation_readids.txt",
    conda:
        "../envs/bam_splitting.yaml"
    shell:
        """
        python3 scripts/splitting_random.py \
            --bam_path {input.bam_path} \
            --output_path outputs/splits/{wildcards.experiment_name}/ \
        """

rule create_split_fast5s:
    '''
    Creates new multiread fast5 files for given readids, so they can be loaded faster during training/inference
    '''
    input:
        ids = "outputs/splits/{experiment_name}/{split}_readids.txt", #The split needs to be non-empty txt file
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(),
    output: #TODO add outputs/splits/expname/{split} folder as output for viz rules
        "outputs/splits/{experiment_name}/FAST5_{split}_SPLIT_DONE.txt",
        directory("outputs/splits/{experiment_name}/{split}"),
    conda:
        "../envs/fast5_splitting.yaml"
    threads: 16
    shell:
        """
        fast5_subset \
            --input {input.experiment_path} \
            --recursive \
            --save_path outputs/splits/{wildcards.experiment_name}/{wildcards.split}/ \
            --threads {threads} \
            --read_id_list {input.ids}
            
        touch {output}
        """
        
rule create_split_files_list:
    input:
        "outputs/splits/{experiment_name}/FAST5_{split}_SPLIT_DONE.txt" #TODO for all split, dont require this - new rule, allow all or scratch compeltely?
    output:
        txt_file="outputs/splits/{experiment_name}/{split}_fast5s_list.txt"
    run:
        files_list = experiments_data[wildcards.experiment_name].get_split_fast5_files(wildcards.split)
        with open(output.txt_file, "w") as out_file:
            for file_path in files_list:
                out_file.write(str(file_path) + "\n")

