
import argparse as ap
import copy
import csv
import glob
from jinja2 import Environment, FileSystemLoader
import json
import numpy as np
import os
import shutil

from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE, GT_REF_HOM
from PipelineConfig import *
from PrintModelReport import createRocImage

#file constants - TODO: consider making some of these options or moving to PipelineConfig
MAIN_TEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/report_templates'
FORMAT_FN = os.path.dirname(os.path.abspath(__file__))+'/format.json'

def loadMetadata():
    '''
    This will load a bunch of sample metadata jsons into a dictionary
    @return - a dictionary where key is a sample number and value is a dictionary of metadata
    '''
    #load the shared metadata file
    md = {}
    for sampleJson in SAMPLE_JSONS:
        fp = open(sampleJson, 'r')
        d = json.load(fp)
        fp.close()

        assert(len(d.keys() & md.keys()) == 0)
        md.update(d)
    return md

def getModelResults(aligner, caller):
    '''
    This will pull ALL the results for an aligner/caller combination and put it in a dictionary to be
    later consumed by Jinja2.
    @param aligner - the aligner to get information for
    @param caller - the caller to get information for
    @return - a dictionary values for use by Jinja2
    '''
    #get all the RTG-specific results for this aligner/caller combo
    rtgResults = getRtgResults(aligner, caller)

    #get all the training results for this aligner/caller combo
    trainingResults = getTrainingResults(aligner, caller)

    retDict = {
        'ALIGNER' : aligner,
        'CALLER' : caller,
        'RTG_RESULTS' : rtgResults,
        'TRAINING_RESULTS' : trainingResults
    }
    return retDict

def getRtgResults(aligner, caller):
    '''
    This will load results specific to the RTG evaluation, things like number of TP, TN, etc.
    @param aligner - the aligner to get information for
    @param caller - the caller to get information for
    @return - a dictionary values for use by Jinja2
    '''
    retDict = {}

    rtgSummaryPattern = '%s/rtg_results/%s/%s/*/summary.txt' % (DATA_DIRECTORY, aligner, caller)
    rtgMetrics = sorted(glob.glob(rtgSummaryPattern))
    retDict['SAMPLE_SUMMARY'] = {}
    totalDict = {}
    for fn in rtgMetrics:
        slid = fn.split('/')[-2]
        summaryDict = loadRtgSummary(fn)

        #make sure we got the big summary
        assert(summaryDict['Threshold'] == "None")
        retDict['SAMPLE_SUMMARY'][slid] = summaryDict

        #add things to the total dict
        for k in summaryDict:
            if k in ['Threshold']:
                pass
            elif k in totalDict:
                totalDict[k].append(summaryDict[k])
            else:
                totalDict[k] = [summaryDict[k]]
    
    for k in totalDict.keys():
        m = np.mean(totalDict[k])
        s = np.std(totalDict[k])
        totalDict[k] = {
            'MEAN' : m,
            'STDEV' : s
        }
    retDict['TOTAL_SUMMARY'] = totalDict

    #get the feature stats
    summaryFN = '%s/pipeline/feature_stats/%s/%s/feature_stats.tsv' % (REPO_DIRECTORY, aligner, caller)
    if os.path.exists(summaryFN):
        featureDict = {}
        fp = open(summaryFN, 'r')
        dictReader = csv.DictReader(fp, delimiter='\t')

        #break them up by variant type and genotype
        varTypes = ['SNV', 'INDEL']
        callTypes = ['HET', 'HOM', 'HE2']

        for d in dictReader:
            sample = d['sample']
            if sample == '' or sample == 'sample':
                continue
            
            if sample not in featureDict:
                featureDict[sample] = {}

            assessType = d['TP/FP']
            totalSum = 0
            for vt in varTypes:
                if vt not in featureDict[sample]:
                    featureDict[sample][vt] = {}
                for ct in callTypes:
                    if ct not in featureDict[sample][vt]:
                        featureDict[sample][vt][ct] = {}
                    featureDict[sample][vt][ct][assessType] = int(d[vt+'-'+ct])
                    totalSum += int(d[vt+'-'+ct])
            
            if 'sum' not in featureDict[sample]:
                featureDict[sample]['sum'] = {}
            featureDict[sample]['sum'][assessType] = totalSum

        fp.close()
        retDict['FEATURES'] = featureDict
    else:
        print('WARNING: could not find feature file "%s"' % summaryFN)

    return retDict

def loadRtgSummary(fn):
    '''
    Loads a summary file into a dictionary
    @param fn - an RTG summary.txt filename
    @return - dict where key is a label and value is the overall results
    '''
    #pull the data
    fp = open(fn, 'r')
    headers = list(filter(lambda x: x != '', fp.readline().rstrip().split(' ')))
    fp.readline()
    bestScores = list(filter(lambda x: x != '', fp.readline().rstrip().split(' ')))
    dataVals = list(filter(lambda x: x != '', fp.readline().rstrip().split(' ')))
    fp.close()
    
    #save it
    if len(dataVals) == 0:
        dataVals = bestScores
    #resultsDict = {h : dataVals[i] for i, h in enumerate(headers)}
    resultsDict = {}
    for i, h in enumerate(headers):
        try:
            val = int(dataVals[i])
        except ValueError:
            try:
                val = float(dataVals[i])
            except ValueError:
                val = dataVals[i]
        resultsDict[h] = val
    return resultsDict

def getTrainingResults(aligner, caller):
    '''
    This will retrieve training results for the specified aligner/caller combo
    @param aligner - the aligner specified
    @param caller - the caller specified
    @return - a dictionary of metrics for this combo
    '''
    #pull the summary results and parse the clinical fragments
    summaryFN = '%s/pipeline/model_summaries/%s/%s/model_summary.tsv' % (REPO_DIRECTORY, aligner, caller)
    clinicalResults = {}
    fp = open(summaryFN, 'r')

    #hunt for the section we care about then TSV parse it
    for l in fp:
        if l.startswith('[clinical_model'):
            pieces = l.rstrip()[1:-1].split(' ')
            assert(pieces[0] == 'clinical_model')
            assert(pieces[1].startswith('min='))
            assert(pieces[2].startswith('tar='))
            clinicalMinimum = float(pieces[1][4:])
            clinicalTarget = float(pieces[2][4:])
            break
    dictReader = csv.DictReader(fp, delimiter='\t')
    for d in dictReader:
        clinicalResults[d['variant_type']] = d
    fp.close()
    
    #now pull out all the ROC curves for the combination
    imageDict = {}
    for vt in VAR_TRANSLATE.keys():
        for gt in GT_TRANSLATE.keys():
            if gt == GT_REF_HOM:
                continue
            
            reformKey = VAR_TRANSLATE[vt]+'_'+GT_TRANSLATE[gt]
            imageFN = '%s/pipeline/model_summaries/%s/%s/roc_curves/%s.png' % (REPO_DIRECTORY, aligner, caller, reformKey)
            if os.path.exists(imageFN):
                imageDict[reformKey] = imageFN
            else:
                imageDict[reformKey] = 'NO_IMAGE_FOUND'

    ret = {
        'CLINICAL_MINIMUM' : clinicalMinimum,
        'CLINICAL_TARGET' : clinicalTarget,
        'CLINICAL_MODELS' : clinicalResults,
        'IMAGE_FILENAMES' : imageDict
    }
    return ret

def generateReport(dataDict, prefix):
    '''
    This will fill in all the template to create our report and then call the appropriate latex commands.
    @param dataDict - the dictionary containing all the data needed to populate our report
    @param prefix - the file path prefix to generate reports at; main report will be [prefix].pdf
    @return - None, but it will make a report using the prefix above
    '''
    texDir = prefix+'_tex'
    if not os.path.exists(texDir):
        os.makedirs(texDir)
    dataDict['PREFIX'] = prefix
    env = Environment(loader=FileSystemLoader(MAIN_TEX_DIR))
    
    #functions we need to pass through
    dataDict['sorted'] = sorted
    #dataDict['enumerate'] = enumerate
    #dataDict['len'] = len

    #build all generic templates
    templateTups = [
        ('supplement_outline.tex', 'supplement_outline.tex'),   #the main template
        ('metadata_template.tex', 'metadata_template.tex'),
        ('pipeline_template.tex', 'pipeline_template.tex'),
        ('training_template.tex', 'training_template.tex'),
    ]

    #build all aligner/caller templates
    for aligner in dataDict['PARSED_DATA']:
        for caller in dataDict['PARSED_DATA'][aligner]:
            templateFN = 'full_results_template.tex'
            template = env.get_template(templateFN)

            #for this one, we only pass the partial dictionary
            partialDict = copy.deepcopy(dataDict['PARSED_DATA'][aligner][caller])
            partialDict['FORMAT'] = dataDict['FORMAT']
            partialDict['sorted'] = sorted
            rendered = template.render(partialDict)
            print(rendered)
            print('END_TEMPLATE (%s, %s): "%s"' % (aligner, caller, templateFN))
            print()
            
            postFix = 'full_results_%s_%s.tex' % (aligner, caller)
            renderFN = '%s/%s' % (texDir, postFix)
            fp = open(renderFN, 'wt+')
            fp.write(rendered)
            fp.close()
    
    for templateFN, postFix in templateTups:
        template = env.get_template(templateFN)
        rendered = template.render(dataDict)
        print(rendered)
        print('END_TEMPLATE: "%s"' % (templateFN, ))
        print()
        
        renderFN = '%s/%s' % (texDir, postFix)
        fp = open(renderFN, 'wt+')
        fp.write(rendered)
        fp.close()
    
    #check if we need to generate the PDF also
    #pdfFN = 
    #pdflatex <tex-prefix>.tex <---- need to run three time to make links/refs & such
    #mv <tex-prefix>.pdf <pdfFN>
    for x in range(0, 3):
        #os.system('cd '+MAIN_TEX_DIR+' && pdflatex '+MAIN_TEX_FN)
        os.system('cd %s && %s supplement_outline.tex' % (texDir, LATEX_PATH))
    shutil.copy('%s/supplement_outline.pdf' % (texDir, ), '%s_results.pdf' % (prefix, ))

if __name__ == "__main__":
    #first set up the arg parser
    DESC="A tool for generating our supplementary document for the false positive prediction results."
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('--l1', dest='label1', default=None, help='the label for the first input (default: same as "oldJson")')
    
    #required main arguments
    p.add_argument('outPrefix', type=str, help='the path-prefix for all output files (ex: "/path/to/final")')

    #parse the arguments
    args = p.parse_args()

    #this file contains all the formatting related stuff
    fp = open(FORMAT_FN, 'r')
    FORMAT_DICT = json.load(fp)
    fp.close()

    #sample metadata
    METADATA_DICT = loadMetadata()

    #things for the event of renaming aligners in the display
    ALIGNER_RENAMING = {
        'clinical_sentieon-201808.07' : 'sentieon-201808.07'
    }
    CALLER_RENAMING = {

    }

    #this will load data from each aligner/caller combination
    PARSED_DATA = {}
    summaryPattern = '%s/pipeline/model_summaries/*/*/model_summary.tsv' % REPO_DIRECTORY
    summaryFiles = sorted(glob.glob(summaryPattern))
    for fn in summaryFiles:
        aligner, caller = fn.split('/')[-3:-1]
        comboData = getModelResults(aligner, caller)
        
        #rename
        comboData['ALIGNER_LABEL'] = ALIGNER_RENAMING.get(aligner, aligner)
        comboData['CALLER_LABEL'] = CALLER_RENAMING.get(caller, caller)

        #save the values
        if aligner not in PARSED_DATA:
            PARSED_DATA[aligner] = {}
        
        #with the renaming of stuff, need to make sure there are no collisions
        assert(caller not in PARSED_DATA[aligner])
        PARSED_DATA[aligner][caller] = comboData
    
    #this dictionary will eventually be passed to the Jinja2 template generator, so all data should go into it
    fullDataDict = {
        'PARSED_DATA' : PARSED_DATA,
        'FORMAT' : FORMAT_DICT,
        'METADATA' : METADATA_DICT,
        'ALIGNER_ORDER' : sorted(PARSED_DATA.keys(), key=lambda al: ALIGNER_RENAMING.get(al, al))
    }
    print(json.dumps(fullDataDict, indent=4, sort_keys=True))

    #exit()
    generateReport(fullDataDict, args.outPrefix)