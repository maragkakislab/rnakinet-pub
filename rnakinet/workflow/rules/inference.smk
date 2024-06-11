rule run_inference:
    input: 
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(),
        model_path = lambda wildcards: models_data[wildcards.model_name].get_path(),
    output: #pickle file for legacy visualization rules, TODO refactor visualizations into using csv
        csv_path = 'outputs/{prediction_type}/{model_name}/{experiment_name}/max_pooling.csv',
        pickle_path ='outputs/{prediction_type}/{model_name}/{experiment_name}/max_pooling.pickle',
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: models_data[wildcards.model_name].get_batch_size(),
        max_len = lambda wildcards: models_data[wildcards.model_name].get_max_len(),
        min_len = lambda wildcards: models_data[wildcards.model_name].get_min_len(),
        skip = lambda wildcards: models_data[wildcards.model_name].get_skip(),
        limit = lambda wildcards: '', #TODO refactor away
    threads: 16 #TODO parametrize
    resources: gpus=1
    wildcard_constraints:
        prediction_type='(predictions|predictions_limited)'
    shell:
        """
        python3 scripts/inference.py \
            --path {input.experiment_path} \
            --checkpoint {input.model_path} \
            --max-workers {threads} \
            --batch-size {params.batch_size} \
            {params.limit} \
            --max-len {params.max_len} \
            --min-len {params.min_len} \
            --skip {params.skip} \
            --csv_output {output.csv_path} \
            --pickle_output {output.pickle_path} \
        """

# run with snakemake --resources parallel_lock=1 to avoid paralelization and multiple runs utilizing the gpu, slowing the time
rule run_inference_speedtest:
    input: 
        experiment_path = lambda wildcards: experiments_data[wildcards.experiment_name].get_path(),
        model_path = lambda wildcards: models_data[wildcards.model_name].get_path(),
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/speedtest/readlimit_{reads_limit}_threads_{threads}.json'
    conda:
        "../envs/inference.yaml"
    params:
        batch_size = lambda wildcards: models_data[wildcards.model_name].get_batch_size(),
        max_len = lambda wildcards: models_data[wildcards.model_name].get_max_len(),
        min_len = lambda wildcards: models_data[wildcards.model_name].get_min_len(),
        skip = lambda wildcards: models_data[wildcards.model_name].get_skip(),
        arch = lambda wildcards: models_data[wildcards.model_name].get_arch(),
        reads_limit = lambda wildcards: wildcards.reads_limit
    threads: lambda wildcards: int(wildcards.threads)
    resources: 
        gpus=1,
        parallel_lock=1, #used to restrict parallelization of this rule
    shell:
        """
        python3 scripts/inference_speedtest.py \
            --arch {params.arch} \
            --path {input.experiment_path} \
            --checkpoint {input.model_path} \
            --max_workers {threads} \
            --batch-size {params.batch_size} \
            --reads_limit {params.reads_limit} \
            --max-len {params.max_len} \
            --min-len {params.min_len} \
            --skip {params.skip} \
            --smake_threads {threads} \
            --exp_name {wildcards.experiment_name} \
            --output {output} \
        """
        
