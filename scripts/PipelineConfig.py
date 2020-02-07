
############################################################
#Core pipeline config
############################################################
#this is the repo directory
REPO_DIRECTORY = '/gpfs/gpfs1/home/jholt/sanger_less_tests'

#root directory for RTG-based analysis, should contain VCFs as well (see README for directions)
DATA_DIRECTORY = '/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline'

############################################################
#Slack config
############################################################
#set to False to disable slack notifications for pipeline completion
ENABLE_SLACK_NOTIFICATIONS = True

#json with sub-dictionary 'urls' where key is a channel name (such as the value in ENABLED_SLACK_CHANNEL) 
# and value is a Slack endpoint URL
ENABLED_SLACK_URLS = '/gpfs/gpfs1/home/jholt/slack_integration/data/slack_urls.json'
ENABLED_SLACK_CHANNEL = "@holtjma"
    