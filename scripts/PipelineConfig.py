
#this is a module we use for storing all our paths systematically, aka a bunch of strings
from csl_pipeline.config import (
    JHOLT_HOME
)

############################################################
#Core pipeline config
############################################################
#this is the repo directory
REPO_DIRECTORY = f'{JHOLT_HOME}/sanger_less_tests'

#a dictionary containing metadata for the samples
SAMPLE_JSONS = [
    #'%s/scripts/GIAB_v1.json' % REPO_DIRECTORY
    #'%s/scripts/GIAB_v2.json' % REPO_DIRECTORY
    f'{REPO_DIRECTORY}/scripts/GIAB_pcrfree.json'
]

#root directory for RTG-based analysis which is expected to conform to a standard directory structure, and contain
# the following files
# {DATA_DIRECTORY}/variant_calls/{aligner}/{caller}/{sample}.vcf.gz - the expected VCF file paths
# {DATA_DIRECTORY}/rtg_results/{aligner}/{caller}/{sample} - folder containing standard RTG results, specifically summary.txt, fp.vcf.gz, and tp.vcf.gz
#DATA_DIRECTORY = '/cluster/home/jholt/csl_validations/core_pipeline_analysis/pipeline'
DATA_DIRECTORY = f'{JHOLT_HOME}/CSL_pipeline_benchmark/pipeline'

#this is where you specify the pipelines to run; every REFERENCE, ALIGNER, and VARIANT_CALLER combo will be run together
#  and each triplet gets it's own model trained for it
REFERENCES = [
    #'clinical'
]

ALIGNERS = [
    #'bwa-mem-0.7.17-BQSR'
]
VARIANT_CALLERS = [
    #'strelka-2.9.10'
]

#if you have specific pipelines (i.e. non-pairs, or maybe externally generated), make the appropriate directory 
# structure (see note above) and hard-code them here
FULL_PIPES = [
    #('dragen-07.011.352.3.2.8b', 'dragen-07.011.352.3.2.8b'), #this is what we originally trained with
    #('dragen-07.021.510.3.5.7', 'dragen-07.021.510.3.5.7'), #this is what follow ups were on
    #('clinical', 'sentieon-201808.07', 'strelka-2.9.10')
    ('hg38_T2T_masked', 'sentieon-202112.01', 'dnascope-1.0-202112.01-PO')
]

#pipeline parameters, thread count currently only applies to the actual model training (CV to be precise)
THREADS_PER_PROC = 33 #was 16 in v1

############################################################
#Documentation config
############################################################

#only necessary if you plan to generate a data summary (i.e. the Supplementary Document)
LATEX_PATH = f'{JHOLT_HOME}/texlive/2019/bin/x86_64-linux/pdflatex'

############################################################
#Slack config
############################################################

#set to False to disable slack notifications for pipeline completion
ENABLE_SLACK_NOTIFICATIONS = True

#json with sub-dictionary 'urls' where key is a channel name (such as the value in ENABLED_SLACK_CHANNEL) 
# and value is a Slack endpoint URL
ENABLED_SLACK_URLS = f'{JHOLT_HOME}/slack_integration/data/slack_urls.json'
ENABLED_SLACK_CHANNEL = "@holtjma"
