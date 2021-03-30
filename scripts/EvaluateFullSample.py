
import argparse as ap
import datetime
import json
import numpy as np
import os
from sklearn.metrics import confusion_matrix

from EvaluateVariants import loadModels, getClinicalModel
from TrainingConfig import *

def runAnalysis(model_directory, featureDir, sample):
    '''
    TODO
    '''
    print(model_directory)
    print(featureDir)
    print(sample)

    #load the models
    stats, models = loadModels(args.model_directory)

    #load the full data
    #TP first
    tpFN = '%s/%s_tp.npy' % (featureDir, sample)
    if os.path.exists(tpFN):
        tpOrder = '%s/%s_tp_fields.json' % (featureDir, sample)
        fp = open(tpOrder, 'r')
        tpFields = json.load(fp)
        fp.close()
        tpVar = np.load(tpFN, 'r')

    #now false positives
    fpFN = '%s/%s_fp.npy' % (featureDir, sample)
    fpOrder = '%s/%s_fp_fields.json' % (featureDir, sample)
    fp = open(fpOrder, 'r')
    fpFields = json.load(fp)
    fp.close()
    fpVar = np.load(fpFN, 'r')

    if os.path.exists(tpFN):
        assert(tpFields == fpFields)
    else:
        tpFields = fpFields
        tpVar = np.zeros(dtype=fpVar.dtype, shape=(0, fpVar.shape[1]))
    
    #constants
    targetRecall = 0.995
    acceptedRecall = 0.99

    for k in stats.keys():
        analyzeResults(k, stats[k], models[k], targetRecall, acceptedRecall, tpFields, tpVar, fpVar)

def analyzeResults(modelType, stats, models, targetRecall, acceptedRecall, fieldsList, tpVals, fpVals):
    #get the clinical model
    clinicalModelDict = getClinicalModel(stats, acceptedRecall, str(targetRecall))
    bestModelName = clinicalModelDict['model_name']
    bestModelTargetRecall = clinicalModelDict['eval_recall']

    if bestModelName == None:
        #print(modelType, 'FAILED')
        variantType = int(modelType.split('_')[0])
        callType = int(modelType.split('_')[1])
        filtersEnabled = True #(acceptedVT != -1 or acceptedGT != -1)
        if filtersEnabled:
            #print('[%s] Filtering variants by type: %s' % (str(datetime.datetime.now()), modelType))
            flInd = fieldsList.index('VAR-TYPE')
            flInd2 = fieldsList.index('CALL-GT')

        if filtersEnabled:
            #figure out which variant match the filter criteria
            tpSearchCrit = (tpVals[:, flInd] == variantType) & (tpVals[:, flInd2] == callType)
            fpSearchCrit = (fpVals[:, flInd] == variantType) & (fpVals[:, flInd2] == callType)
            
            #now extract them and replace them
            tpVals = tpVals[tpSearchCrit, :]
            fpVals = fpVals[fpSearchCrit, :]

        print(modelType, 'FAILED', 'N/A', 1.0, 1.0, fpVals.shape[0], tpVals.shape[0], sep='\t')
    else:
        #print(modelType, bestModelName, bestModelTargetRecall)
        modelTargetRecall = bestModelTargetRecall
        #evalList = [bestModelName]
        variantType = int(modelType.split('_')[0])
        callType = int(modelType.split('_')[1])

        #make sure all feature sets are identical and set up the additional field also
        rawFeatures = models[bestModelName]['FEATURES']
        coreFeatureNames = [tuple(f.split('-')) for f in models[bestModelName]['FEATURES']]
        #print(coreFeatureNames)
        acceptedVT = models[bestModelName]['FILTER_VAR_TYPE']
        acceptedGT = models[bestModelName]['FILTER_CALL_TYPE']
        filtersEnabled = True #(acceptedVT != -1 or acceptedGT != -1)

        #if the filters are enabled, we need to make sure both were filtered; or stuff will break! :O
        if filtersEnabled:
            assert(acceptedVT != -1 and acceptedGT != -1)
        
        REMOVED_LABELS = []
        
        if filtersEnabled:
            #print('[%s] Filtering variants by type: %s' % (str(datetime.datetime.now()), modelType))
            flInd = fieldsList.index('VAR-TYPE')
            flInd2 = fieldsList.index('CALL-GT')

        #match w/e the models have for fields even if present in our data
        for label in fieldsList:
            if label not in rawFeatures:
                REMOVED_LABELS.append(label)

        if len(REMOVED_LABELS) > 0:
            #get the indices to remove and update features appropriately
            removedIndices = [fieldsList.index(rfl) for rfl in REMOVED_LABELS]
            assert(-1 not in removedIndices)
            featureLabels = [v for i, v in enumerate(fieldsList) if (i not in removedIndices)]
        
        if filtersEnabled:
            #figure out which variant match the filter criteria
            tpSearchCrit = (tpVals[:, flInd] == variantType) & (tpVals[:, flInd2] == callType)
            fpSearchCrit = (fpVals[:, flInd] == variantType) & (fpVals[:, flInd2] == callType)
            
            #now extract them and replace them
            tpVals = tpVals[tpSearchCrit, :]
            fpVals = fpVals[fpSearchCrit, :]
        
        #if we have labels to remove, strip them out here
        if len(REMOVED_LABELS) > 0:
            tpVals = np.delete(tpVals, removedIndices, 1)
            fpVals = np.delete(fpVals, removedIndices, 1)
        
        #we aren't doing a full test, so cut the input sizes down
        if False and USE_SUBSET:
            tpVals = tpVals[:SUBSET_SIZE]
            fpVals = fpVals[:SUBSET_SIZE]

        #now we need to pull out final train/test sets
        combined_X = np.vstack([tpVals, fpVals])
        combined_Y = np.array([1]*tpVals.shape[0] + [0]*fpVals.shape[0])
        if FLIP_TP:
            combined_Y = 1 - combined_Y
        
        #do the prediction
        clf = models[bestModelName]['MODEL']
        thresh = stats[bestModelName]['ALL_SUMMARY'][modelTargetRecall]['TRAIN_THRESHOLD'][0]
        y_pred_prob = clf.predict_proba(combined_X)[:, 1]
        adjPred = [1 if y >= thresh else 0 for y in y_pred_prob]
        #print(adjPred[tpVals.shape[0]:])
        
        #extra data
        runs = []
        c = 0
        for x in range(tpVals.shape[0], len(adjPred)):
            if adjPred[x] == 0:
                c += 1
            else:
                if c > 1:
                    runs.append(c)
                    '''
                    print(c)
                    for v in combined_X[x-c:x]:
                        print('\t', v)
                    '''
                c = 0
        if c > 1:
            runs.append(c)

        adjConf = confusion_matrix(combined_Y, adjPred)
        test_FPR = adjConf[0, 1] / np.sum(adjConf[0, :])
        test_TPR = adjConf[1, 1] / np.sum(adjConf[1, :])

        print(modelType, bestModelName, bestModelTargetRecall, test_FPR, test_TPR, fpVals.shape[0], tpVals.shape[0], runs, sep='\t')

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for evaluating variants for false positive status"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-c', '--codicem', dest='codicem', default=None, help='a Codicem CSV file with variants to evaluate (default: None)')
    #p.add_argument('-v', '--variants', dest='variants', default=None, help='variant coordinates to evaluate (default: None)')
    #p.add_argument('-m', '--model', dest='model', default='best', help='the model name to use (default: best)')
    #p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')
    #p.add_argument('-o', '--output', dest='outFN', default=None, help='the place to send output to (default: stdout)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    #p.add_argument('sample_vcf', type=str, help='VCF file with variants to evaulate')
    p.add_argument('feature_directory', type=str, help='directory with the sample features')
    p.add_argument('sample_id', type=str, help='the sample identifier to pull known TP and FP calls')

    #parse the arguments
    args = p.parse_args()

    runAnalysis(args.model_directory, args.feature_directory, args.sample_id)