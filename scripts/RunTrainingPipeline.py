
import argparse as ap
import csv
import datetime
import json
import os
import requests
import subprocess

#disable by setting to None
#SLACK_CHANNEL = None
#SLACK_CHANNEL = "#csl-pipeline"
SLACK_CHANNEL = "@holtjma"

SNAKEFILE_PATH = os.path.dirname(os.path.realpath(__file__))+'/TrainingPipeline.smk'
SNAKEMAKE_CLUSTER_CONFIG = os.path.dirname(os.path.realpath(__file__))+'/cluster.json'

def parseSlids(slidStr):
    '''
    This will parse a list of SL##s 
    @param slidStr - a string containing comma separated slids; "-" is also accepted for ranges; 
        ex: "SL123456-SL123467,SL333333"; it can also be a filename with same info on lines
    @return - a list of all slids we need to pull data for
    '''
    ret = []
    if os.path.exists(slidStr):
        #load a file with lines
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
            assert(subFrags[0][0:2] == 'SL')
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
    urlJson = '/gpfs/gpfs1/home/jholt/slack_integration/data/slack_urls.json'
    fp = open(urlJson, 'r')
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
    DESC="Wrapper script for running the pipelines"
    DEFAULT_JSON = '/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/scripts/sample_metadata.json'
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-t', '--train-models', dest='train_models', action='store_true', default=False, help='train the models using the pipeline structure (default: False)')
    p.add_argument('-x', '--execute', dest='execute', action='store_true', default=False, help='execute the commands (default: False)')

    #required main arguments
    p.add_argument('slids', type=str, help='the list of slids separate by commas (ex: "SL123456-SL123467,SL333333")')
    
    #parse the arguments
    args = p.parse_args()
    
    snakemakeFrags = [
        'snakemake',
        '--snakefile', SNAKEFILE_PATH,
        '--cluster-config', SNAKEMAKE_CLUSTER_CONFIG,
        '--cluster', '"bsub -o {cluster.log} -J {cluster.name} -n {threads} -M {cluster.memory} -R \\"span[hosts=1] rusage[mem={cluster.memory}]\\""',
        '-j', '5000',
        '--config', 'sampleData="%s"' % (args.slids, ),
        '-p', #always print commands
        '-k', #keep going in the event of partial failure
    ]

    if not args.execute:
        print('Dry run mode: enabled')
        snakemakeFrags.append('-n') #dry-run mode
    print()
    
    buildFrags = []
    somethingToDo = False

    if args.train_models:
        somethingToDo = True
        buildFrags.append('train_models')

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
        print('WARNING: No snakemake objects were specified to be generated.  Exiting without doing any work.')
    
