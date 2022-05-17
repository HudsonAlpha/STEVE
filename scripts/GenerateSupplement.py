
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

def getModelResults(datasets, pipeline, reference, aligner, caller):
    '''
    This will pull ALL the results for an aligner/caller combination and put it in a dictionary to be
    later consumed by Jinja2.
    @param pipeline - the pipeline subdirectory
    @param reference - the reference to get information for
    @param aligner - the aligner to get information for
    @param caller - the caller to get information for
    @return - a dictionary values for use by Jinja2
    '''
    #get all the RTG-specific results for this aligner/caller combo
    rtgResults = getRtgResults(datasets, pipeline, reference, aligner, caller)

    #get all the training results for this aligner/caller combo
    trainingResults = getTrainingResults(pipeline, reference, aligner, caller)
    strictResults = getTrainingResults(pipeline, reference, aligner, caller, True)
    eli5Results = getEli5Results(pipeline, reference, aligner, caller)

    retDict = {
        'ALIGNER' : aligner,
        'CALLER' : caller,
        'RTG_RESULTS' : rtgResults,
        'TRAINING_RESULTS' : trainingResults,
        'STRICT_RESULTS' : strictResults,
        'ELI5_RESULTS' : eli5Results
    }
    return retDict

def getRtgResults(datasets, pipeline, reference, aligner, caller):
    '''
    This will load results specific to the RTG evaluation, things like number of TP, TN, etc.
    @param pipeline - the pipeline subdirectory (usually "pipeline")
    @param reference - the reference to get information for
    @param aligner - the aligner to get information for
    @param caller - the caller to get information for
    @return - a dictionary values for use by Jinja2
    '''
    retDict = {}

    rtgSummaryPattern = '%s/rtg_results/%s/%s/%s/*/summary.txt' % (DATA_DIRECTORY, reference, aligner, caller)
    rtgMetrics = sorted(glob.glob(rtgSummaryPattern))
    retDict['SAMPLE_SUMMARY'] = {}
    totalDict = {}
    for fn in rtgMetrics:
        slid = fn.split('/')[-2]
        if slid not in datasets:
            continue
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
    summaryFN = '%s/%s/feature_stats/%s/%s/%s/feature_stats.tsv' % (REPO_DIRECTORY, pipeline, reference, aligner, caller)
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

def getTrainingResults(pipeline, reference, aligner, caller, strict=False):
    '''
    This will retrieve training results for the specified aligner/caller combo
    @param pipeline - the pipeline subdirectory
    @param reference - the reference specified
    @param aligner - the aligner specified
    @param caller - the caller specified
    @param strict - if True, load the strict version of the results
    @return - a dictionary of metrics for this combo
    '''
    #pull the summary results and parse the clinical fragments
    if not strict:
        summaryFN = '%s/%s/model_summaries/%s/%s/%s/model_summary.tsv' % (REPO_DIRECTORY, pipeline, reference, aligner, caller)
    else:
        summaryFN = '%s/%s/model_summaries/%s/%s/%s/strict_summary.tsv' % (REPO_DIRECTORY, pipeline, reference, aligner, caller)
    clinicalResults = {}
    fp = open(summaryFN, 'r')

    #hunt for the section we care about then TSV parse it
    var_counts = {}
    for l in fp:
        if l.startswith('[clinical_model'):
            pieces = l.rstrip()[1:-1].split(' ')
            assert(pieces[0] == 'clinical_model')
            if pieces[1].startswith('min=') and pieces[2].startswith('tar='):
                clinicalMinimum = float(pieces[1][4:])
                clinicalTarget = float(pieces[2][4:])
            elif pieces[1].startswith('target_global_precision='):
                clinicalMinimum = 'dynamic'
                clinicalTarget = 'dynamic'
            else:
                raise Exception('unknown clinical_model format')
            break
        elif l.startswith('['):
            vt, tpcount, fpcount = l.rstrip()[1:-1].split(' ')
            assert(tpcount[0:3] == 'TP=')
            assert(fpcount[0:3] == 'FP=')
            tpcount = int(tpcount[3:])
            fpcount = int(fpcount[3:])
            var_counts[vt] = {
                'TP' : tpcount,
                'FP' : fpcount
            }

    dictReader = csv.DictReader(fp, delimiter='\t')
    for d in dictReader:
        d = dict(d)

        '''
        #if we decide CI is useful, this would add it
        if d['best_model'] != 'None':
            #calculate confidence intervals; TP counts goes with FPR/TP flag rate
            z = 1.96 #95% CI
            tpcount = var_counts[d['variant_type']]['TP']
            fpr = float(d['final_FPR'])
            fprCI = z * np.sqrt(fpr * (1-fpr) / tpcount)
            d['final_FPR'] = '%0.4f+-%0.4f' % (fpr, fprCI)

            #FP counts goes with recall/sensitivity/capture rate
            fpcount = var_counts[d['variant_type']]['FP']
            recall = float(d['final_recall'])
            recallCI = z * np.sqrt(recall * (1-recall) / fpcount)
            d['final_recall'] = '%0.4f+-%0.4f' % (recall, recallCI)
        '''
        
        clinicalResults[d['variant_type']] = d
    fp.close()
    
    #now pull out all the ROC curves for the combination
    imageDict = {}
    for vt in VAR_TRANSLATE.keys():
        for gt in GT_TRANSLATE.keys():
            if gt == GT_REF_HOM:
                continue
            
            reformKey = VAR_TRANSLATE[vt]+'_'+GT_TRANSLATE[gt]
            imageFN = '%s/%s/model_summaries/%s/%s/%s/roc_curves/%s.png' % (REPO_DIRECTORY, pipeline, reference, aligner, caller, reformKey)
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

def getEli5Results(pipeline, reference, aligner, caller):
    '''
    This will retrieve ELI5 results if available
    @param pipeline - the pipeline subdirectory
    @param aligner - the aligner specified
    @param caller - the caller specified
    @return - a dictionary of metrics for this combo
    '''
    jsonFN = '%s/%s/eli5_summaries/%s/%s/%s/model_eli5.json' % (REPO_DIRECTORY, pipeline, reference, aligner, caller)

    #catch the situation where we didn't run ELI5 for w/e reason
    ret = {}
    if not os.path.exists(jsonFN):
        dataDict = {}
    else:
        fp = open(jsonFN, 'r')
        dataDict = json.load(fp)
        fp.close()
    
    cumulativeFI = {}

    for vt in VAR_TRANSLATE.keys():
        for gt in GT_TRANSLATE.keys():
            if gt == GT_REF_HOM:
                continue
            reformKey = VAR_TRANSLATE[vt]+'_'+GT_TRANSLATE[gt]
            if reformKey in dataDict:
                #include errors if they occur and feature importances
                bm = dataDict[reformKey]['best_model']
                featImp = dataDict[reformKey].get('eli5', {}).get('feature_importances', {})
                if featImp == None:
                    featImp = []
                else:
                    featImp = featImp.get('importances', [])
                if bm != None:
                    ret[reformKey] = {
                        'BEST_MODEL' : bm,
                        'ERROR' : dataDict[reformKey].get('eli5', {}).get('error', None),
                        'FEATURE_IMPORTANCES' : featImp
                    }

                    #create the cumulative dictionary here
                    for featDict in featImp:
                        featureName = featDict['feature']
                        if featureName not in cumulativeFI:
                            cumulativeFI[featureName] = {}
                        cumulativeFI[featureName][reformKey] = featDict
                else:
                    ret[reformKey] = {
                        'BEST_MODEL' : 'None',
                        'ERROR' : 'No passing models.',
                        'FEATURE_IMPORTANCES' : []
                    }
            else:
                ret[reformKey] = {
                    'BEST_MODEL' : 'None',
                    'ERROR' : 'No ELI5 results detected',
                    'FEATURE_IMPORTANCES' : []
                }
    
    #now post-process the cumulative
    for fn in cumulativeFI:
        cumWeight = 0
        for reformKey in cumulativeFI[fn]:
            cumWeight += cumulativeFI[fn][reformKey]['weight']
        cumulativeFI[fn]['CUMULATIVE_WEIGHT'] = cumWeight
    
    orderedCumWeight = sorted(cumulativeFI.keys(), key=lambda k: cumulativeFI[k]['CUMULATIVE_WEIGHT'], reverse=True)
    ret['COMBINED_DICT'] = cumulativeFI
    ret['COMBINED_ORDER'] = orderedCumWeight

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
    dataDict['REGION_COUNT_IMAGE'] = f'{MAIN_TEX_DIR}/region_count.png'
    dataDict['IGV_REGION_IMAGE'] = f'{MAIN_TEX_DIR}/igv_region.png'
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
        ('hardcoded_template.tex', 'hardcoded_template.tex'),
    ]

    #build all aligner/caller templates
    for aligner in dataDict['PARSED_DATA']:
        for caller in dataDict['PARSED_DATA'][aligner]:
            templateFN = 'full_results_template.tex'
            template = env.get_template(templateFN)

            #for this one, we only pass the partial dictionary
            partialDict = copy.deepcopy(dataDict['PARSED_DATA'][aligner][caller])
            partialDict['FORMAT'] = dataDict['FORMAT']
            partialDict['METADATA'] = dataDict['METADATA']
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
    DESC="A tool for generating our supplementary document for the false positive prediction results"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-p', '--pipeline-subdir', dest='pipeline', default='pipeline', help='the subdirectory containing the outputs (default: "pipeline"')
    
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
    combos = []
    for aligner in ALIGNERS:
        for caller in VARIANT_CALLERS:
            combos.append((aligner, caller))
    for fullPipe in FULL_PIPES:
        combos.append(fullPipe)

    for reference, aligner, caller in combos:
        #check for the summary file
        summaryFN = '%s/%s/model_summaries/%s/%s/%s/model_summary.tsv' % (REPO_DIRECTORY, args.pipeline, reference, aligner, caller)
        if not os.path.exists(summaryFN):
            print(f'WARNING: Summary file for {reference}/{aligner}/{caller} was not found, re-run pipeline?')
            continue

        #get the results
        comboData = getModelResults(METADATA_DICT, args.pipeline, reference, aligner, caller)
        
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