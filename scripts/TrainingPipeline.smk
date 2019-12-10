
import os

from RunTrainingPipeline import parseSlids

#Constants for the validated pipeline
PIPELINE_DIRECTORY = '/gpfs/gpfs1/home/jholt/sanger_less_tests/pipeline'
EXTRACT_SCRIPT = os.path.dirname(os.path.realpath(PIPELINE_DIRECTORY))+'/scripts/ExtractFeatures.py'
TRAINING_SCRIPT = os.path.dirname(os.path.realpath(PIPELINE_DIRECTORY))+'/scripts/TrainModels.py'
ALIGNERS = [
    #'bwa-mem-0.7.17',
    'bwa-mem-0.7.17-BQSR',
    'sentieon-201808.07'
]
VARIANT_CALLERS = [
    'strelka-2.9.10'
]
FULL_PIPES = [
    ('dragen-07.011.352.3.2.8b', 'dragen-07.011.352.3.2.8b')
]
THREADS_PER_PROC = 16

#parse the sample names for the next steps
RAW_SLIDS = config['sampleData']
SAMPLE_LIST = sorted(set(parseSlids(RAW_SLIDS)))

def getTrainedModels():
    '''
    This will return a list of all models we expect to generate given the above parameters
    '''
    ret = []
    for al in ALIGNERS:
        for vc in VARIANT_CALLERS:
            #ret.append('%s/trained_models/%s/%s/model_metadata.json' % (PIPELINE_DIRECTORY, al, vc))
            ret.append('%s/trained_models/%s/%s/models.p' % (PIPELINE_DIRECTORY, al, vc))
    
    for al, vc in FULL_PIPES:
        ret.append('%s/trained_models/%s/%s/models.p' % (PIPELINE_DIRECTORY, al, vc))

    return ret

rule train_models:
    input:
        *getTrainedModels()

##############################################################################################
#FEATURE EXTRACTION
##############################################################################################

rule ExtractFeatures:
    input:
        rawVcf="/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/variant_calls/{aligner}/{caller}/{slid}.vcf.gz",
        tpVcf="/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/rtg_results/{aligner}/{caller}/{slid}/tp.vcf.gz",
        fpVcf="/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/rtg_results/{aligner}/{caller}/{slid}/fp.vcf.gz"
    output:
        fp_features="{pipeline_dir}/features/{aligner}/{caller}/{slid}_fp.npy",
        fp_fields="{pipeline_dir}/features/{aligner}/{caller}/{slid}_fp_fields.json",
        tp_features="{pipeline_dir}/features/{aligner}/{caller}/{slid}_tp.npy",
        tp_fields="{pipeline_dir}/features/{aligner}/{caller}/{slid}_tp_fields.json"
    params:
        script=EXTRACT_SCRIPT,
        prefix="{pipeline_dir}/features"
    log: "{pipeline_dir}/logs/features/{aligner}/{caller}/{slid}.log"
    threads: 1 #this currently isn't multi-threaded in any way
    shell:
        '''
        python3 -u {params.script} \
            {wildcards.aligner} \
            {wildcards.caller} \
            {wildcards.slid} \
            {params.prefix}
        '''

##############################################################################################
#TRAINING
##############################################################################################

def getFeatureFiles(wildcards):
    '''
    This will return the set of feature files required to perform training
    '''
    pd = wildcards['pipeline_dir']
    al = wildcards['aligner']
    vc = wildcards['caller']
    ret = []
    postFixes = [
        'fp_fields.json',
        'fp.npy',
        'tp_fields.json',
        'tp.npy'
    ]
    for slid in SAMPLE_LIST:
        for pf in postFixes:
            ret.append('%s/features/%s/%s/%s_%s' % (pd, al, vc, slid, pf))
    return ret

rule TrainModels:
    input:
        getFeatureFiles
    output:
        #metadata is now captured in the model pickle file
        #metadata="{pipeline_dir}/trained_models/{aligner}/{caller}/model_metadata.json",
        models="{pipeline_dir}/trained_models/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{aligner}/{caller}/stats.json"
    params:
        script=TRAINING_SCRIPT,
        feature_dir="{pipeline_dir}/features/{aligner}/{caller}",
        slids=RAW_SLIDS,
        output_prefix="{pipeline_dir}/trained_models/{aligner}/{caller}"
    log: "{pipeline_dir}/logs/trained_models/{aligner}/{caller}_training.log"
    threads: THREADS_PER_PROC #only applies to CV, but still a big speedup
    shell:
        '''
        python3 -u {params.script} \
            --split-by-type \
            -p {threads} \
            {params.feature_dir} \
            {params.slids} \
            {params.output_prefix}
        '''