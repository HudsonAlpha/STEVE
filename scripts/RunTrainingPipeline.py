
import argparse as ap
import csv
import datetime
import json
import os
import requests
import subprocess

from PipelineConfig import ENABLE_SLACK_NOTIFICATIONS, ENABLED_SLACK_CHANNEL, ENABLED_SLACK_URLS

#disable by setting to None
if ENABLE_SLACK_NOTIFICATIONS:
    SLACK_CHANNEL = ENABLED_SLACK_CHANNEL
else:
    SLACK_CHANNEL = None

SNAKEFILE_PATH = os.path.dirname(os.path.realpath(__file__))+'/TrainingPipeline.smk'
SNAKEMAKE_CLUSTER_CONFIG = os.path.dirname(os.path.realpath(__file__))+'/cluster.json'
SNAKEMAKE_PROFILE = "lsf"

def parseSlids(slidStr):
    '''
    This will parse a list of SL##s 
    @param slidStr - a JSON file, plain text file, or raw string containing sample identifier 
        raw string ex: "SL123467,SL333333"; if using SL###, then ranges can be used such as "SL123456-SL123467"
        plain text ex: can be any of the raw string fields, one per line
        JSON ex: see GIAB_all.json for an example
    @return - a list of all sample identifiers we need to pull data for
    '''
    ret = []
    if os.path.exists(slidStr):
        if slidStr.endswith('.json'):
            #assume a dictionary where each key is a sample
            fp = open(slidStr, 'r')
            j = json.load(fp)
            fp.close()
            frags = []
            for k in j.keys():
                frags.append(k)
        else:
            #load a file with one sample number per line
            fp = open(slidStr, 'rt')
            frags = []
            for l in fp:
                frags.append(l.rstrip())
            fp.close()
    else:
        #load as fragments
        frags = slidStr.split(',')

    for fragment in frags:
        subFrags = fragment.split('-')
        assert(len(subFrags) <= 2)
        if len(subFrags) == 1:
            #single slid
            #assert(subFrags[0][0:2] == 'SL')
            ret.append(subFrags[0])
        elif len(subFrags) == 2:
            #range of slids
            r1 = subFrags[0]
            r2 = subFrags[1]
            assert(r1[0:2] == 'SL' and r2[0:2] == 'SL')
            i1 = int(r1[2:])
            i2 = int(r2[2:])
            assert(i2 >= i1)
            for i in range(i1, i2+1):
                ret.append('SL'+str(i))
        else:
            raise Exception('How did you get here?')
    return ret

def sendSlackMessage(channel, message):
    '''
    This is basically a wrapper for sending a slack message
    @param channel - the channel to send output to
    @param message - a string indicating what should be sent to the channel
    '''
    fp = open(ENABLED_SLACK_URLS, 'r')
    j = json.load(fp)
    fp.close()
    
    myUrl = j['urls'][channel]
    print('Sending slack message to "%s" : "%s"' % (channel, message))
    MAX_RETRY = 10
    currTry = 0
    while currTry < MAX_RETRY:
        currTry += 1
        try:
            response = requests.post(myUrl, json={"text" : message})
            if response.ok:
                break
            else:
                print('Failed to send slack message, received response: "%s"', response.text)
        except Exception as e:
            print('Error while trying to send slack message:')
            print(e)
        
        print('Waiting 10 seconds to retry...')
        time.sleep(10)
        print('Retrying...')

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Wrapper script for running the full training pipeline"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-d', '--data-stats', dest='data_stats', action='store_true', default=False, help='get data statistics on features (default: False)')
    p.add_argument('-e', '--model-eli5', dest='model_eli5', action='store_true', default=False, help='get eli5 results for "clinical" models (default: False)')
    p.add_argument('-s', '--summarize-models', dest='summarize_models', action='store_true', default=False, help='summarize model results (default: False)')
    p.add_argument('-t', '--train-models', dest='train_models', action='store_true', default=False, help='train the models using the pipeline structure (default: False)')
    p.add_argument('-u', '--unlock', dest='unlock', action='store_true', default=False, help='unlock the directory in the event of snakemake failure (default: False)')
    p.add_argument('-x', '--execute', dest='execute', action='store_true', default=False, help='execute the commands (default: False)')

    #required main arguments
    p.add_argument('slids', type=str, help='sample labels (.json, .txt, comma-separated entry)')
    
    #parse the arguments
    args = p.parse_args()
    
    #This pipeline assumes snakemake is installed and that it's running on an LSF cluster
    snakemakeFrags = [
        'snakemake',
        '--profile', SNAKEMAKE_PROFILE,
        '--snakefile', SNAKEFILE_PATH,
        '--cluster-config', SNAKEMAKE_CLUSTER_CONFIG,
        #'--cluster', '"bsub -o {cluster.log} -J {cluster.name} -n {threads} -M {cluster.memory} -R \\"span[hosts=1] rusage[mem={cluster.memory}]\\""',
        #'-j', '5000',
        '--config', 'sampleData="%s"' % (args.slids, ),
        '-p', #always print commands
        '-k', #keep going in the event of partial failure
    ]

    if not args.execute:
        print('Dry run mode: enabled')
        snakemakeFrags.append('-n') #dry-run mode
    print()

    if args.unlock:
        print('Unlock: enabled')
        snakemakeFrags.append('--unlock')
        buildFrags.append('--unlock')
    print()
    
    buildFrags = []
    somethingToDo = False

    if args.train_models:
        somethingToDo = True
        buildFrags.append('train_models')

    if args.summarize_models:
        somethingToDo = True
        buildFrags.append('summarize_models')
    
    if args.data_stats:
        somethingToDo = True
        buildFrags.append('data_stats')
    
    if args.model_eli5:
        somethingToDo = True
        buildFrags.append('model_eli5')

    if somethingToDo:
        fullCmd = ' '.join(snakemakeFrags+buildFrags)
        print('Executing: ')
        print(fullCmd)
        returnCode = os.system(fullCmd)
        if args.execute:
            slackFrags = [
                '<!channel>',
                'Sangerless training pipeline - finished running with return code "%s":' % (returnCode, ),
                'Build SLIDs: '+args.slids,
                'Build Targets: '+str(buildFrags)
            ]
            if SLACK_CHANNEL != None:
                sendSlackMessage(SLACK_CHANNEL, '\n'.join(slackFrags))
            else:
                print('\n'.join(slackFrags))
    else:
        print('WARNING: No snakemake objects were specified to be generated.  Exiting without doing any work.')
    
