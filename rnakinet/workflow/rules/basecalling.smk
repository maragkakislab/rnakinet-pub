# BASECALLER_VERSION = 'ont-guppy_6.4.8_linux64'
# DORADO_VERSION = 'dorado-0.5.3-linux-x64'

guppy_location = 'ont-guppy/bin/guppy_basecaller'
dorado_location = 'dorado-0.5.3-linux-x64/bin/dorado'

# Downloads the basecaller software
rule get_basecaller:
    output: guppy_location
    shell:
        f"""
        wget https://cdn.oxfordnanoportal.com/software/analysis/ont-guppy_6.4.8_linux64.tar.gz
        tar -xf ont-guppy_6.4.8_linux64.tar.gz
        """

rule get_dorado:
    output: dorado_location
    shell:
        f"""
        wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-linux-x64.tar.gz
        tar -xf dorado-0.5.3-linux-x64.tar.gz
        """


        
# Basecalls fast5 files into fastq files
rule basecalling:
    input: 
        experiment_path = lambda wildcards: directory(experiments_data[wildcards.experiment_name].get_path()),
        basecaller_location = guppy_location, 
    output:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    params:
        kit = lambda wildcards: experiments_data[wildcards.experiment_name].get_kit(),
        flowcell = lambda wildcards: experiments_data[wildcards.experiment_name].get_flowcell(),
    threads: 32
    resources: gpus=1
    shell:
        """
        {input.basecaller_location} \
            -x "auto" \
            --kit {params.kit} \
            --flowcell {params.flowcell} \
            --records_per_fastq 0 \
            --trim_strategy none \
            --save_path outputs/basecalling/{wildcards.experiment_name}/guppy/ \
            --recursive \
            --gpu_runners_per_device 1 \
            --num_callers {threads} \
            --chunks_per_runner 512 \
            --compress_fastq \
            --calib_detect \
            --input_path {input.experiment_path} \
            
        echo {input.experiment_path} > {output}
        """
        
#TODO make rule for dorado download dorado download --model rna004_130bps_hac@v3.0.1
rule basecalling_dorado:
    input: 
        pod5_file = lambda wildcards: f'outputs/basecalling/{wildcards.experiment_name}/pod5/output.pod5',
        basecaller_location = dorado_location,
    output:
        done_txt ='outputs/basecalling/{experiment_name}/DONE_dorado.txt',
        out_reads = 'outputs/basecalling/{experiment_name}/dorado/all_reads.fastq'
    params:
        kit = lambda wildcards: experiments_data[wildcards.experiment_name].get_kit(),
        flowcell = lambda wildcards: experiments_data[wildcards.experiment_name].get_flowcell(),
    threads: 32
    resources: gpus=1
    shell:
        """
        {input.basecaller_location} basecaller rna004_130bps_hac@v3.0.1 {input.pod5_file} \
            --no-trim \
            --emit-fastq \
            > {output.out_reads}

        echo {input.pod5_file} > {output}
        """

            
rule zip_dorado_basecalls:
    input:
        'outputs/basecalling/{experiment_name}/dorado/all_reads.fastq'
    output:
        'outputs/basecalling/{experiment_name}/dorado/all_reads.fastq.gz'
    shell:
        'gzip -c {input} > {output}'
        
            
rule convert_fast5_to_pod5:
    input:
        experiment_path = lambda wildcards: directory(experiments_data[wildcards.experiment_name].get_path()),
    output:
        'outputs/basecalling/{experiment_name}/pod5/output.pod5',
    conda:
        '../envs/pod5_convert.yaml'
    shell:
        'pod5 convert fast5 {input.experiment_path}/*.fast5 --output {output}'
    

# Merges multiple fastq files into a single fastq file
rule merge_fastq_files:
    input:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    output:
        "outputs/basecalling/{experiment_name}/guppy/reads.fastq.gz"
    conda:
        "../envs/merge_fastq.yaml"
    shell:
        """
        zcat outputs/basecalling/{wildcards.experiment_name}/guppy/pass/fastq_runid*.fastq.gz | pigz > {output}
        """
        
        
rule merge_fastq_files_fail_pass:
    input:
        'outputs/basecalling/{experiment_name}/DONE.txt'
    output:
        "outputs/basecalling/{experiment_name}/guppy/all_reads.fastq.gz"
    conda:
        "../envs/merge_fastq.yaml"
    shell:
        """
        zcat outputs/basecalling/{wildcards.experiment_name}/guppy/*/fastq_runid*.fastq.gz | pigz > {output}
        """