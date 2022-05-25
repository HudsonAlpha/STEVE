
import os

from RunTrainingPipeline import parseSlids
from PipelineConfig import *
from TrainingConfig import GLOBAL_AUTO_TARGET_PRECISION

#derived from repo 
PIPELINE_DIRECTORY = '%s/pipeline_pcrfree' % REPO_DIRECTORY
#CONDA_ENV = '%s/scripts/conda_config.yaml' % REPO_DIRECTORY
EXTRACT_SCRIPT = '%s/scripts/ExtractFeatures.py' % REPO_DIRECTORY
TRAINING_SCRIPT = '%s/scripts/TrainModels.py' % REPO_DIRECTORY
DATA_REPORT_SCRIPT = '%s/scripts/PrintDataReport.py' % REPO_DIRECTORY
MODEL_REPORT_SCRIPT = '%s/scripts/PrintModelReport.py' % REPO_DIRECTORY
MODEL_ELI5_SCRIPT = '%s/scripts/ExtractELI5Results.py' % REPO_DIRECTORY

#parse the sample names for the next steps
RAW_SLIDS = config['sampleData']
SAMPLE_LIST = sorted(set(parseSlids(RAW_SLIDS)))

def getTrainedModels():
    '''
    This will return a list of all models we expect to generate given the above parameters
    '''
    ret = []
    
    for ref in REFERENCES:
        for al in ALIGNERS:
            for vc in VARIANT_CALLERS:
                ret.append('%s/trained_models/%s/%s/models.p' % (PIPELINE_DIRECTORY, ref, al, vc))
    
    for ref, al, vc in FULL_PIPES:
        ret.append('%s/trained_models/%s/%s/%s/models.p' % (PIPELINE_DIRECTORY, ref, al, vc))

    return ret

def getModelSummaries():
    '''
    This will return a list of model summaries we expect at this end of this
    '''
    ret = []
    for ref in REFERENCES:
        for al in ALIGNERS:
            for vc in VARIANT_CALLERS:
                ret.append('%s/model_summaries/%s/%s/%s/model_summary.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))
                ret.append('%s/model_summaries/%s/%s/%s/strict_summary.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))
    
    for ref, al, vc in FULL_PIPES:
        ret.append('%s/model_summaries/%s/%s/%s/model_summary.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))
        ret.append('%s/model_summaries/%s/%s/%s/strict_summary.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))

    return ret

def getDataStats():
    '''
    This will return a list of data stat files we expect at the end of this
    '''
    ret = []
    for ref in REFERENCES:
        for al in ALIGNERS:
            for vc in VARIANT_CALLERS:
                ret.append('%s/feature_stats/%s/%s/%s/feature_stats.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))
    
    for ref, al, vc in FULL_PIPES:
        ret.append('%s/feature_stats/%s/%s/%s/feature_stats.tsv' % (PIPELINE_DIRECTORY, ref, al, vc))

    return ret

def getModelEli5s():
    '''
    This will return a list of model summaries we expect at this end of this
    '''
    ret = []
    for ref in REFERENCES:
        for al in ALIGNERS:
            for vc in VARIANT_CALLERS:
                ret.append('%s/eli5_summaries/%s/%s/%s/model_eli5.json' % (PIPELINE_DIRECTORY, ref, al, vc))
    
    for ref, al, vc in FULL_PIPES:
        ret.append('%s/eli5_summaries/%s/%s/%s/model_eli5.json' % (PIPELINE_DIRECTORY, ref, al, vc))

    return ret

rule train_models:
    input:
        *getTrainedModels()

rule summarize_models:
    input:
        *getModelSummaries()

rule data_stats:
    input:
        *getDataStats()

rule model_eli5:
    input:
        *getModelEli5s()

##############################################################################################
#FEATURE EXTRACTION
##############################################################################################

rule ExtractFeatures:
    input:
        rawVcf="%s/variant_calls/{reference}/{aligner}/{caller}/{slid}.vcf.gz" % DATA_DIRECTORY,
        tpVcf="%s/rtg_results/{reference}/{aligner}/{caller}/{slid}/tp.vcf.gz" % DATA_DIRECTORY,
        fpVcf="%s/rtg_results/{reference}/{aligner}/{caller}/{slid}/fp.vcf.gz" % DATA_DIRECTORY
    output:
        fp_features="{pipeline_dir}/features/{reference}/{aligner}/{caller}/{slid}_fp.npy",
        fp_fields="{pipeline_dir}/features/{reference}/{aligner}/{caller}/{slid}_fp_fields.json",
        tp_features="{pipeline_dir}/features/{reference}/{aligner}/{caller}/{slid}_tp.npy",
        tp_fields="{pipeline_dir}/features/{reference}/{aligner}/{caller}/{slid}_tp_fields.json"
    params:
        script=EXTRACT_SCRIPT,
        prefix="{pipeline_dir}/features"
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/features/{reference}/{aligner}/{caller}/{slid}.log"
    threads: 1 #this currently isn't multi-threaded in any way
    shell:
        '''
        python3 -u {params.script} \
            {wildcards.reference} \
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
    @param wildcards - the wildcards from the snakemake rule; need "pipeline_dir", "aligner", and "caller"
    @return - a list of feature files required for training, see "postFixes" variable for specifics
    '''
    #get the wildcards
    pd = wildcards['pipeline_dir']
    ref = wildcards['reference']
    al = wildcards['aligner']
    vc = wildcards['caller']
    
    #file post-fixes that we need to make sure are gathered up
    postFixes = [
        'fp_fields.json',
        'fp.npy',
        'tp_fields.json',
        'tp.npy'
    ]
    
    #now do it
    ret = []
    for slid in SAMPLE_LIST:
        for pf in postFixes:
            ret.append('%s/features/%s/%s/%s/%s_%s' % (pd, ref, al, vc, slid, pf))
    return ret

rule TrainModels:
    input:
        getFeatureFiles
    output:
        models="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/stats.json"
    params:
        script=TRAINING_SCRIPT,
        feature_dir="{pipeline_dir}/features/{reference}/{aligner}/{caller}",
        slids=RAW_SLIDS,
        output_prefix="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}"
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/trained_models/{reference}/{aligner}/{caller}_training.log"
    threads: THREADS_PER_PROC #only applies to CV, but still a big speedup
    resources:
        mem_mb=128000
    shell:
        '''
        python3 -u {params.script} \
            --split-by-type \
            -p {threads} \
            {params.feature_dir} \
            {params.slids} \
            {params.output_prefix}
        '''

##############################################################################################
#Statistics gathering
##############################################################################################

rule SummarizeFeatures:
    input:
        features=getFeatureFiles
    output:
        summary="{pipeline_dir}/feature_stats/{reference}/{aligner}/{caller}/feature_stats.tsv"
    params:
        script=DATA_REPORT_SCRIPT,
        prefix="{pipeline_dir}/features/{reference}/{aligner}/{caller}"
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/feature_stats/{reference}/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            {params.prefix} > \
            {output.summary}
        '''

rule SummarizeModels:
    input:
        models="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/stats.json"
    output:
        summary="{pipeline_dir}/model_summaries/{reference}/{aligner}/{caller}/model_summary.tsv",
        images=directory("{pipeline_dir}/model_summaries/{reference}/{aligner}/{caller}/roc_curves")
    params:
        script=MODEL_REPORT_SCRIPT,
        prefix="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}",
        global_precision=GLOBAL_AUTO_TARGET_PRECISION
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/model_summaries/{reference}/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            -r {output.images} \
            -g {params.global_precision} \
            {params.prefix} > \
            {output.summary}
        '''

rule StrictSummarizeModels:
    input:
        models="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/stats.json"
    output:
        summary="{pipeline_dir}/model_summaries/{reference}/{aligner}/{caller}/strict_summary.tsv"
    params:
        script=MODEL_REPORT_SCRIPT,
        prefix="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}"
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/model_summaries/{reference}/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            -m 0.999 \
            -t 1.0 \
            {params.prefix} > \
            {output.summary}
        '''

rule ModelEli5:
    input:
        models="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}/stats.json"
    output:
        eli5="{pipeline_dir}/eli5_summaries/{reference}/{aligner}/{caller}/model_eli5.json"
    params:
        script=MODEL_ELI5_SCRIPT,
        global_precision=GLOBAL_AUTO_TARGET_PRECISION,
        prefix="{pipeline_dir}/trained_models/{reference}/{aligner}/{caller}"
    #conda:
    #    CONDA_ENV
    log: "{pipeline_dir}/logs/eli5_summaries/{reference}/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            -g {params.global_precision} \
            {params.prefix} \
            {output.eli5}
        '''
