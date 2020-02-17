
import os

from RunTrainingPipeline import parseSlids
from PipelineConfig import DATA_DIRECTORY, REPO_DIRECTORY

#derived from repo 
PIPELINE_DIRECTORY = '%s/pipeline' % REPO_DIRECTORY
EXTRACT_SCRIPT = '%s/scripts/ExtractFeatures.py' % REPO_DIRECTORY
TRAINING_SCRIPT = '%s/scripts/TrainModels.py' % REPO_DIRECTORY
DATA_REPORT_SCRIPT = '%s/scripts/PrintDataReport.py' % REPO_DIRECTORY
MODEL_REPORT_SCRIPT = '%s/scripts/PrintModelReport.py' % REPO_DIRECTORY

ALIGNERS = [
    'bwa-mem-0.7.17-BQSR',
    #'sentieon-201808.07'
]
VARIANT_CALLERS = [
    'strelka-2.9.10'
]
FULL_PIPES = [
    ('dragen-07.011.352.3.2.8b', 'dragen-07.011.352.3.2.8b'),
    ('clinical_sentieon-201808.07', 'strelka-2.9.10')
]
THREADS_PER_PROC = 16

#parse the sample names for the next steps
RAW_SLIDS = config['sampleData']
SAMPLE_LIST = sorted(set(parseSlids(RAW_SLIDS))))

def getTrainedModels():
    '''
    This will return a list of all models we expect to generate given the above parameters
    '''
    ret = []
    for al in ALIGNERS:
        for vc in VARIANT_CALLERS:
            ret.append('%s/trained_models/%s/%s/models.p' % (PIPELINE_DIRECTORY, al, vc))
    
    for al, vc in FULL_PIPES:
        ret.append('%s/trained_models/%s/%s/models.p' % (PIPELINE_DIRECTORY, al, vc))

    return ret

def getModelSummaries():
    '''
    This will return a list of model summaries we expect at this end of this
    '''
    ret = []
    for al in ALIGNERS:
        for vc in VARIANT_CALLERS:
            ret.append('%s/model_summaries/%s/%s/model_summary.tsv' % (PIPELINE_DIRECTORY, al, vc))
            ret.append('%s/model_summaries/%s/%s/strict_summary.tsv' % (PIPELINE_DIRECTORY, al, vc))
    
    for al, vc in FULL_PIPES:
        ret.append('%s/model_summaries/%s/%s/model_summary.tsv' % (PIPELINE_DIRECTORY, al, vc))
        ret.append('%s/model_summaries/%s/%s/strict_summary.tsv' % (PIPELINE_DIRECTORY, al, vc))

    return ret

def getDataStats():
    '''
    This will return a list of data stat files we expect at the end of this
    '''
    ret = []
    for al in ALIGNERS:
        for vc in VARIANT_CALLERS:
            ret.append('%s/feature_stats/%s/%s/feature_stats.tsv' % (PIPELINE_DIRECTORY, al, vc))
    
    for al, vc in FULL_PIPES:
        ret.append('%s/feature_stats/%s/%s/feature_stats.tsv' % (PIPELINE_DIRECTORY, al, vc))

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

##############################################################################################
#FEATURE EXTRACTION
##############################################################################################

rule ExtractFeatures:
    input:
        rawVcf="%s/variant_calls/{aligner}/{caller}/{slid}.vcf.gz" % DATA_DIRECTORY,
        tpVcf="%s/rtg_results/{aligner}/{caller}/{slid}/tp.vcf.gz" % DATA_DIRECTORY,
        fpVcf="%s/rtg_results/{aligner}/{caller}/{slid}/fp.vcf.gz" % DATA_DIRECTORY
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

##############################################################################################
#Statistics gathering
##############################################################################################

rule SummarizeFeatures:
    input:
        features=getFeatureFiles
    output:
        summary="{pipeline_dir}/feature_stats/{aligner}/{caller}/feature_stats.tsv"
    params:
        script=DATA_REPORT_SCRIPT,
        prefix="{pipeline_dir}/features/{aligner}/{caller}"
    log: "{pipeline_dir}/logs/feature_stats/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            {params.prefix} > \
            {output.summary}
        '''

rule SummarizeModels:
    input:
        models="{pipeline_dir}/trained_models/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{aligner}/{caller}/stats.json"
    output:
        summary="{pipeline_dir}/model_summaries/{aligner}/{caller}/model_summary.tsv",
        images=directory("{pipeline_dir}/model_summaries/{aligner}/{caller}/roc_curves")
    params:
        script=MODEL_REPORT_SCRIPT,
        prefix="{pipeline_dir}/trained_models/{aligner}/{caller}"
    log: "{pipeline_dir}/logs/model_summaries/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            -r {output.images} \
            {params.prefix} > \
            {output.summary}
        '''

rule StrictSummarizeModels:
    input:
        models="{pipeline_dir}/trained_models/{aligner}/{caller}/models.p",
        rocs="{pipeline_dir}/trained_models/{aligner}/{caller}/rocs.json",
        stats="{pipeline_dir}/trained_models/{aligner}/{caller}/stats.json"
    output:
        summary="{pipeline_dir}/model_summaries/{aligner}/{caller}/strict_summary.tsv"
    params:
        script=MODEL_REPORT_SCRIPT,
        prefix="{pipeline_dir}/trained_models/{aligner}/{caller}"
    log: "{pipeline_dir}/logs/model_summaries/{aligner}/{caller}.log"
    threads: 1
    shell:
        '''
        python3 -u {params.script} \
            -m 0.999 \
            -t 1.0 \
            {params.prefix} > \
            {output.summary}
        '''