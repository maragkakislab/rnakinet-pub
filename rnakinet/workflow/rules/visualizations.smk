rule create_distribution_plot:
    input:
        files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=pos_neg_pairs[wildcards.pair_name]['negatives']+pos_neg_pairs[wildcards.pair_name]['positives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
    output:
        'outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(boxplot|violin)'
    conda:
        '../envs/visual.yaml'
    params:
        exp_names = lambda wildcards: pos_neg_pairs[wildcards.pair_name]['negatives']+pos_neg_pairs[wildcards.pair_name]['positives']
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --files {input.files} \
            --output {output} \
            --model-name {wildcards.model_name} \
            --exp-names {params.exp_names} \
        """

rule create_classification_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
            experiment_name=[exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,),
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
    output:
        'outputs/visual/predictions/{model_name}/{group}_{pooling}_pooling_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve|pr_ratios)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = lambda wildcards: [exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['negatives']],
        pos_experiments = lambda wildcards: [exp for pair_name in comparison_groups[wildcards.group] for exp in pos_neg_pairs[pair_name]['positives']],
        neg_group_names = lambda wildcards: [pair_name for pair_name in comparison_groups[wildcards.group] for _ in pos_neg_pairs[pair_name]['negatives']],
        pos_group_names = lambda wildcards: [pair_name for pair_name in comparison_groups[wildcards.group] for _ in pos_neg_pairs[pair_name]['positives']],    
        chosen_threshold = lambda wildcards: models_data[wildcards.model_name].get_threshold(),
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name {wildcards.model_name} \
        """
        
rule create_classification_plot_nanoid_model:
    input:
        neg_files = 'outputs/predictions/nanoid/nanoid_hsa_dRNA_HeLa_DMSO_1_complete.pickle',
        pos_files = 'outputs/predictions/nanoid/hsa_dRNA_HeLa_5EU_polyA_REL5_2_nanoid_complete.pickle',
    output:
        'outputs/visual/predictions/NANOID/DMSO_1_REL5_2_{plot_type}.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve|pr_ratios)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = ['hsa_dRNA_HeLa_DMSO_1'],
        pos_experiments = ['dRNA_HeLa_5EU_polyA_REL5_2'],
        neg_group_names = ['NIA_HELA'],
        pos_group_names = ['NIA_HELA'],    
        chosen_threshold = 0.5,
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name NANOIDMODEL \
        """
        
        
rule create_chromosome_plot_nanoid_model:
    input:
        neg_files = 'outputs/predictions/nanoid/nanoid_hsa_dRNA_HeLa_DMSO_1_complete.pickle',
        pos_files = 'outputs/predictions/nanoid/hsa_dRNA_HeLa_5EU_polyA_REL5_2_nanoid_complete.pickle',
        neg_bams = 'outputs/alignment/20220520_hsa_dRNA_HeLa_DMSO_1/reads-align.genome.sorted.bam',
        pos_bams = 'outputs/alignment/20220303_hsa_dRNA_HeLa_5EU_polyA_REL5_2/reads-align.genome.sorted.bam',
    output:
        'outputs/visual/predictions/NANOID/DMSO_1_REL5_2_chrplot_{plot_type}.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        train_chrs = [str(i) for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','Y','MT']],
        valid_chrs = [str(i) for i in [20]],
        test_chrs = [str(i) for i in [1]],
        chosen_threshold = 0.5,
        ylim = 1.0,
    wildcard_constraints:
        plot_type='(auroc|f1)'
    shell:
        """
        python3 scripts/chrwise_plot.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --train_chrs {params.train_chrs} \
            --valid_chrs {params.valid_chrs} \
            --test_chrs {params.test_chrs} \
            --plot_type {wildcards.plot_type} \
            --chosen_threshold {params.chosen_threshold} \
            --ylim {params.ylim} \
            --output {output} \
        """
        
        
rule create_models_classification_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
            experiment_name=pos_neg_pairs[wildcards.pair_name]['negatives'],
            model_name=model_comparison_groups[wildcards.model_group],
            pooling=wildcards.pooling,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=pos_neg_pairs[wildcards.pair_name]['positives'],
            model_name=model_comparison_groups[wildcards.model_group],
            pooling=wildcards.pooling,
        ),
    output:
        'outputs/visual/predictions/multimodel/{model_group}/{pair_name}_{pooling}_pooling_{plot_type}_multi.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)',
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = lambda wildcards: [exp for model_name in model_comparison_groups[wildcards.model_group] for exp in pos_neg_pairs[wildcards.pair_name]['negatives']],
        pos_experiments = lambda wildcards: [exp for model_name in model_comparison_groups[wildcards.model_group] for exp in pos_neg_pairs[wildcards.pair_name]['positives']],
        neg_group_names = lambda wildcards: [model_name for model_name in model_comparison_groups[wildcards.model_group] for _ in pos_neg_pairs[wildcards.pair_name]['negatives']],
        pos_group_names = lambda wildcards: [model_name for model_name in model_comparison_groups[wildcards.model_group] for _ in pos_neg_pairs[wildcards.pair_name]['positives']],
        chosen_threshold = 0.0, #Not defined
        model_name = 'Undefined',
    shell:
        """
        python3 scripts/{wildcards.plot_type}.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name {params.model_name} \
        """
        

rule create_chromosome_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']],
            model_name=wildcards.model_name, 
            pooling=wildcards.pooling,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']]),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']]),
    output:
        'outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_chr_{plot_type}.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        train_chrs = [str(i) for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,'X','Y','MT']],
        valid_chrs = [str(i) for i in [20]],
        test_chrs = [str(i) for i in [1]],
        chosen_threshold = lambda wildcards: models_data[wildcards.model_name].get_threshold(),
        ylim = 0.80,
    wildcard_constraints:
        plot_type='(auroc|f1)'
    shell:
        """
        python3 scripts/chrwise_plot.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --train_chrs {params.train_chrs} \
            --valid_chrs {params.valid_chrs} \
            --test_chrs {params.test_chrs} \
            --plot_type {wildcards.plot_type} \
            --chosen_threshold {params.chosen_threshold} \
            --ylim {params.ylim} \
            --output {output} \
        """
        
rule create_separated_auroc_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        neg_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['negatives']]),
        pos_bams = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam', experiment_name=[exp for exp in pos_neg_pairs[wildcards.pair_name]['positives']]),
    output:
        'outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf'
    conda:
        '../envs/visual.yaml'
    params:
        chosen_threshold = lambda wildcards: models_data[wildcards.model_name].get_threshold(),
    wildcard_constraints:
        plot_type='(Uperc|length)'
    shell:
        """
        python3 scripts/separated_auroc.py \
            --positives_bams {input.pos_bams} \
            --negatives_bams {input.neg_bams} \
            --positives_predictions {input.pos_files} \
            --negatives_predictions {input.neg_files} \
            --plot_type {wildcards.plot_type} \
            --chosen_threshold {params.chosen_threshold} \
            --output {output} \
        """        

rule create_volcano_plot:
    input:
        "outputs/diff_exp/{time_group}/DESeq_output.tab"
    output:
        "outputs/visual/diff_exp/{time_group}/{column}.pdf"
    conda:
        "../envs/visual.yaml"
    shell:
        """
        python3 scripts/volcano.py \
            --table-path {input} \
            --save-path {output} \
            --column {wildcards.column} \
        """

rule calculate_decay:
    input:
        gene_predictions = 'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_predictions.tsv',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        tl = lambda wildcards: experiments_data[wildcards.experiment_name].get_time(),
    shell:
        """
        python3 scripts/calculate_decay.py \
            --gene-predictions {input.gene_predictions} \
            --tl {params.tl} \
            --output {output} \
        """

rule filter_decay_preds:
    input:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions.tsv',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        min_reads=100,
        max_measured_halflife=None,
        read_cols = ['reads'],
    shell:
        """
        python3 scripts/filter_transcripts.py \
            --table {input} \
            --min-reads {params.min_reads} \
            --max-measured-halflife {params.max_measured_halflife} \
            --read-cols {params.read_cols} \
            --output {output} \
        """
        
rule normalize_decay:
    input:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean.tsv',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean_normalized.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        halflife_column = 'pred_t5'
    shell:
        """
        python3 scripts/normalize_decay.py \
            --table {input} \
            --halflife-column {params.halflife_column} \
            --output {output} \
        """

ref_map = {
    'gene': 'Gene stable ID',
    'transcript': 'Transcript stable ID',
}
rule join_predicted_and_measured_tables:
    input:
        predicted_halflives='outputs/{prediction_type}/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean_normalized.tsv',
        measured_halflives = lambda wildcards: experiments_data[wildcards.experiment_name].get_halflives_name_to_file()[wildcards.halflives_name],
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        table_a_column=lambda wildcards: ref_map[wildcards.reference_level],
        table_b_column=lambda wildcards: wildcards.reference_level,
        halflife_column='t5',
    shell:
        """
        python3 scripts/join_tables.py \
            --table-a {input.predicted_halflives} \
            --table-b {input.measured_halflives} \
            --table-a-column '{params.table_a_column}' \
            --table-b-column '{params.table_b_column}' \
            --halflife-column '{params.halflife_column}' \
            --output {output} \
        """

rule filter_bad_transcripts:
    input:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join.tsv',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join_clean.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        min_reads=100,
        max_measured_halflife=5,
        read_cols = ['reads'],
    shell:
        """
        python3 scripts/filter_transcripts.py \
            --table {input} \
            --min-reads {params.min_reads} \
            --max-measured-halflife {params.max_measured_halflife} \
            --read-cols {params.read_cols} \
            --output {output} \
        """
        
rule normalize_measured_decay:
    input:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join_clean.tsv',
    output:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join_clean_normalized.tsv',
    conda:
        "../envs/visual.yaml"
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        halflife_column = 't5'
    shell:
        """
        python3 scripts/normalize_decay.py \
            --table {input} \
            --halflife-column {params.halflife_column} \
            --output {output} \
        """
    
def get_measured_hl_column(normalization):
    if(normalization == 'unnormalized'):
        return 't5'
    else:
        return f'{normalization}_t5'
    
rule plot_halflives_correlation:
    input:
        'outputs/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_{pooling}_pooling_{reference_level}_level_halflives_join_clean_normalized.tsv'
    output:
        'outputs/visual/{prediction_type}/{model_name}/{experiment_name}/{halflives_name}_halflives_{pooling}_pooling_{reference_level}_{normalization}_decay_plot.pdf'
    conda:
        "../envs/visual.yaml" 
    wildcard_constraints:
        prediction_type='(joined_predictions|predictions)'
    params:
        x_column=lambda wildcards: get_measured_hl_column(wildcards.normalization),
        y_column=lambda wildcards: f'{wildcards.normalization}_pred_t5',
        x_label=lambda wildcards: f'{wildcards.normalization} t5 measured',
        y_label=lambda wildcards: f'{wildcards.normalization} t5 predicted',
    shell:
        """
        python3 scripts/correlation_plot.py \
            --table {input} \
            --x-column {params.x_column} \
            --y-column {params.y_column} \
            --x-label '{params.x_label}' \
            --y-label '{params.y_label}' \
            --output {output} \
        """

rule create_decay_read_limit_plot:
    input:
        gene_predictions_list = lambda wildcards: expand('outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_level_predictions.tsv',
                                                        experiment_name=exp_groups[wildcards.group_name],
                                                        model_name=wildcards.model_name,
                                                        pooling=wildcards.pooling,
                                                        reference_level=wildcards.reference_level,),
        gene_halflifes_list = lambda wildcards: [experiments_data[experiment_name].get_halflives_name_to_file()[wildcards.halflives_name] for experiment_name in exp_groups[wildcards.group_name]], 
    output:
        'outputs/visual/predictions/{model_name}/decay/{halflives_name}_halflives_{group_name}_{pooling}_pooling_{reference_level}_read_limit_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        tl_list = lambda wildcards: [experiments_data[experiment_name].get_time() for experiment_name in exp_groups[wildcards.group_name]],
        exp_name_list = lambda wildcards: exp_groups[wildcards.group_name],
    shell:
        """
        python3 scripts/decay_read_limit_plot_multi.py \
            --gene-predictions-list {input.gene_predictions_list} \
            --gene-halflifes-list {input.gene_halflifes_list} \
            --gene-halflifes-gene-column {wildcards.reference_level} \
            --tl-list {params.tl_list} \
            --exp-name-list {params.exp_name_list} \
            --output {output} \
        """
        
#TODO redundant?
rule filter_bad_transcripts_selfcorr:
    input:
        'outputs/predictions/{model_name}/decay/{pooling}_pooling_{reference_level}_level_halflives_join.tsv',
    output:
        'outputs/predictions/{model_name}/decay/{pooling}_pooling_{reference_level}_level_halflives_join_clean.tsv',
    conda:
        "../envs/visual.yaml"
    params:
        min_reads=100,
        max_measured_halflife=None,
        read_cols = ['reads_x','reads_y'],
    shell:
        """
        python3 scripts/filter_transcripts.py \
            --table {input} \
            --min-reads {params.min_reads} \
            --max-measured-halflife {params.max_measured_halflife} \
            --read-cols {params.read_cols} \
            --output {output} \
        """
        
rule create_self_corr_decay_plot:
    input:
        'outputs/predictions/{model_name}/decay/{pooling}_pooling_{reference_level}_level_halflives_join_clean.tsv',
    output:
        'outputs/visual/predictions/{model_name}/self_corr_{pooling}_pooling_{reference_level}_{normalization}_decay_plot.pdf'
    conda:
        "../envs/visual.yaml"
    params:
        x_column=lambda wildcards: f'{wildcards.normalization}_pred_t5_x',
        y_column=lambda wildcards: f'{wildcards.normalization}_pred_t5_y',
        x_label=lambda wildcards: f'{wildcards.normalization} t5 replicate 1',
        y_label=lambda wildcards: f'{wildcards.normalization} t5 replicate 2',
    shell:
        """
        python3 scripts/correlation_plot.py \
            --table {input} \
            --x-column {params.x_column} \
            --y-column {params.y_column} \
            --x-label '{params.x_label}' \
            --y-label '{params.y_label}' \
            --share-axes \
            --output {output} \
        """
        
self_corr_exp_1 = '20230706_mmu_dRNA_3T3_5EU_400_1'
self_corr_exp_2 = '20230816_mmu_dRNA_3T3_5EU_400_2'
rule join_predicted_tables:
    input:
        table_a = 'outputs/predictions/{model_name}/'+self_corr_exp_1+'/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean_normalized.tsv',
        table_b = 'outputs/predictions/{model_name}/'+self_corr_exp_2+'/{pooling}_pooling_{reference_level}_level_halflives_predictions_clean_normalized.tsv',
    output:
        'outputs/predictions/{model_name}/decay/{pooling}_pooling_{reference_level}_level_halflives_join.tsv',
    conda:
        "../envs/visual.yaml"
    params:
        table_a_column=lambda wildcards: ref_map[wildcards.reference_level],
        table_b_column=lambda wildcards: ref_map[wildcards.reference_level],
        halflife_column='pred_t5',
    shell:
        """
        python3 scripts/join_tables.py \
            --table-a {input.table_a} \
            --table-b {input.table_b} \
            --table-a-column '{params.table_a_column}' \
            --table-b-column '{params.table_b_column}' \
            --halflife-column '{params.halflife_column}' \
            --output {output} \
        """

#TODO refactor to use correlation plotting
rule create_fc_plot:
    input:
        gene_level_preds_control=lambda wildcards: expand(
            "outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=condition_control_pairs[wildcards.time_group]['controls'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        gene_level_preds_condition=lambda wildcards: expand(
            "outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=condition_control_pairs[wildcards.time_group]['conditions'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        deseq_output = 'outputs/diff_exp/{time_group}/DESeq_output.tab',
    output:
        "outputs/visual/predictions/{model_name}/{time_group}/{pooling}_pooling_fc_{pred_col}.pdf"
    conda:
        "../envs/visual.yaml"
    params:
        target_col = 'log2FoldChange',
    shell:
        """
        python3 scripts/fc_limit_plot.py \
            --gene-level-preds-control {input.gene_level_preds_control} \
            --gene-level-preds-condition {input.gene_level_preds_condition} \
            --deseq-output {input.deseq_output} \
            --pred-col {wildcards.pred_col} \
            --target-col {params.target_col} \
            --output {output} \
        """
        
rule create_fc_plot_old:
    input:
        gene_level_preds_control=lambda wildcards: expand(
            "outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=condition_control_pairs[wildcards.time_group]['controls'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        gene_level_preds_condition=lambda wildcards: expand(
            "outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling_gene_level_predictions.tsv", 
            experiment_name=condition_control_pairs[wildcards.time_group]['conditions'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
        deseq_output = 'outputs/diff_exp/{time_group}/DESeq_output.tab',
    output:
        "outputs/visual/predictions/{model_name}/{time_group}/{pooling}_pooling_fcold_{pred_col}.pdf"
    conda:
        "../envs/visual.yaml"
    params:
        target_col = 'log2FoldChange',
        min_reads = 100,
    shell:
        """
        python3 scripts/fc_plot.py \
            --gene-level-preds-control {input.gene_level_preds_control} \
            --gene-level-preds-condition {input.gene_level_preds_condition} \
            --deseq-output {input.deseq_output} \
            --pred-col {wildcards.pred_col} \
            --target-col {params.target_col} \
            --min-reads {params.min_reads} \
            --output {output} \
        """
        
# run with --resources parallel_lock=1 to avoid paralelization and multiple runs utilizing the gpu, slowing the time
rule create_inference_speed_plot:
    input:
        jsons = lambda wildcards: expand('outputs/predictions/{model_name}/{experiment_name}/speedtest/readlimit_{reads_limit}_threads_{threads}.json',
                experiment_name=wildcards.experiment_name,
                model_name=wildcards.model_name,
                reads_limit=[10000, 100000, 500000, 1000000],
                threads=wildcards.threads,
        )
    output:
        'outputs/visual/predictions/{model_name}/{experiment_name}/speedtest_threads_{threads}.pdf'
    conda:
        "../envs/visual.yaml"
    shell:
        """
        python3 scripts/speedtest_plot.py \
            --jsons {input.jsons} \
            --output {output} \
        """ 

rule create_datastats:
    input:
        bam_files = lambda wildcards: expand('outputs/alignment/{experiment_name}/reads-align.genome.sorted.bam',
                experiment_name=datastats_groups[wildcards.group],
        ),
    output:
        sizes_output = 'outputs/visual/datastats/{group}_sizes_stats.csv',
        lengths_output = 'outputs/visual/datastats/{group}_lengths_dist.pdf',
    conda:
        "../envs/visual.yaml"
    params:
        experiment_names = lambda wildcards: datastats_groups[wildcards.group]
    shell:
        """
        python3 scripts/datastats.py \
            --bam_files {input.bam_files} \
            --experiment_names {params.experiment_names} \
            --sizes_output {output.sizes_output} \
            --lengths_output {output.lengths_output} \
            --group {wildcards.group} \
        """ 
        
rule create_classification_ratios_plot:
    input:
        neg_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle',
            experiment_name=pos_neg_pairs[wildcards.pair_name]['negatives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,),
        pos_files = lambda wildcards: expand(
            'outputs/predictions/{model_name}/{experiment_name}/{pooling}_pooling.pickle', 
            experiment_name=pos_neg_pairs[wildcards.pair_name]['positives'],
            model_name=wildcards.model_name,
            pooling=wildcards.pooling,
        ),
    output:
        'outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}_ratios.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = lambda wildcards: pos_neg_pairs[wildcards.pair_name]['negatives'],
        pos_experiments = lambda wildcards: pos_neg_pairs[wildcards.pair_name]['positives'],
        neg_group_names = lambda wildcards: wildcards.pair_name,
        pos_group_names = lambda wildcards: wildcards.pair_name,    
        chosen_threshold = lambda wildcards: models_data[wildcards.model_name].get_threshold(),
    shell:
        """
        python3 scripts/{wildcards.plot_type}_ratios.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name {wildcards.model_name} \
        """

rule create_classification_ratios_plot_nanoid_model:
    input:
        neg_files = 'outputs/predictions/nanoid/nanoid_hsa_dRNA_HeLa_DMSO_1_complete.pickle',
        pos_files = 'outputs/predictions/nanoid/hsa_dRNA_HeLa_5EU_polyA_REL5_2_nanoid_complete.pickle',
    output:
        'outputs/visual/predictions/NANOID/DMSO_1_REL5_2_{plot_type}_ratios.pdf'
    wildcard_constraints:
        plot_type='(auroc|thresholds|pr_curve)'
    conda:
        '../envs/visual.yaml'
    params:
        neg_experiments = ['hsa_dRNA_HeLa_DMSO_1'],
        pos_experiments = ['dRNA_HeLa_5EU_polyA_REL5_2'],
        neg_group_names = ['NIA_HELA'],
        pos_group_names = ['NIA_HELA'],    
        chosen_threshold = 0.5,
    shell:
        """
        python3 scripts/{wildcards.plot_type}_ratios.py \
            --positives-in-order {input.pos_files} \
            --negatives-in-order {input.neg_files} \
            --positives-names-in-order {params.pos_experiments} \
            --negatives-names-in-order {params.neg_experiments} \
            --negatives-groups-in-order {params.neg_group_names} \
            --positives-groups-in-order {params.pos_group_names} \
            --output {output} \
            --chosen_threshold {params.chosen_threshold} \
            --model-name NANOIDMODEL \
        """
        
rule create_all_plots:
    input:
        expand("outputs/visual/diff_exp/{time_group}/{column}.pdf",
            time_group=condition_control_pairs.keys(),
            column=['padj','pvalue'],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_chr_{plot_type}.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            pair_name=pos_neg_pairs.keys(),
            plot_type=['auroc','f1'],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf',
            pair_name=[pair_name for pair_name in pos_neg_pairs.keys()],
            model_name = wildcards.model_name, 
            pooling=pooling,
            plot_type=['Uperc', 'length'],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{group}_{pooling}_pooling_{plot_type}.pdf',
            group=[group for group in comparison_groups.keys()],
            model_name = wildcards.model_name, 
            pooling=pooling,
            plot_type=['auroc', 'thresholds','pr_curve'],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{pair_name}_{pooling}_pooling_{plot_type}.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            pair_name=[pair_name for pair_name in pos_neg_pairs.keys()],
            plot_type=['violin'],    
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{experiment_name}/{halflives_name}_halflives_{pooling}_pooling_{reference_level}_{normalization}_{plot_type}.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            experiment_name=exp_groups['mouse_decay_exps'],
            plot_type=['decay_plot'],
            reference_level=['gene','transcript'],
            halflives_name=[key for experiment_name in exp_groups['mouse_decay_exps'] for key in experiments_data[experiment_name].get_halflives_name_to_file().keys()],
            normalization=['unnormalized','standardized','robust','quantile','minmax'],
        ),
        # lambda wildcards: expand('outputs/visual/predictions/{model_name}/{experiment_name}/{pooling}_pooling_{reference_level}_{plot_type}.pdf',
        #     model_name = wildcards.model_name, 
        #     pooling=pooling,
        #     experiment_name=exp_groups['hela_decay_exps'],
        #     plot_type=['decay_plot'],
        #     reference_level=['transcript'],
        # ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/decay/{halflives_name}_halflives_{group_name}_{pooling}_pooling_{reference_level}_{plot_type}.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            group_name=['mouse_decay_exps'],
            plot_type=['read_limit_decay_plot'],
            reference_level=['gene','transcript'],
            halflives_name=[key for experiment_name in exp_groups['mouse_decay_exps'] for key in experiments_data[experiment_name].get_halflives_name_to_file().keys()],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/{time_group}/{pooling}_pooling_fc_{pred_col}.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            time_group=condition_control_pairs.keys(),
            pred_col=['average_score','percentage_modified'],
        ),
        lambda wildcards: expand('outputs/visual/predictions/{model_name}/self_corr_{pooling}_pooling_{reference_level}_{normalization}_decay_plot.pdf',
            model_name = wildcards.model_name, 
            pooling=pooling,
            reference_level=['gene','transcript'],
            normalization=['unnormalized','standardized','robust','quantile','minmax'],
        ),        
    output:
        'outputs/visual/predictions/{model_name}/{pooling}_ALL_DONE.txt'
    shell:
        """
        touch {output}
        """
        
        