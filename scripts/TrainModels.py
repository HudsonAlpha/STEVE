
#typical imports
import argparse as ap
import bisect
import datetime
import json
from json import JSONEncoder
import numpy as np
import pickle

#learning related imports
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import shuffle
from skopt import BayesSearchCV

#custom imports
from ExtractFeatures import VAR_SNP, VAR_INDEL, GT_REF_HET, GT_ALT_HOM, GT_HET_HET, GT_REF_HOM, DEFAULT_MISSING
from RunTrainingPipeline import parseSlids

#config goes in here
from TrainingConfig import *

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        #in the event you have to add these in the future: model the np.float32 instance for single value types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return obj.item()
        return JSONEncoder.default(self, obj)

def trainModels(featureDir, slids, outPrefix, splitByType, numProcs):
    '''
    Trains models given a selection of features
    @param featureDir - the directory containing features
    @param slids - a file with one sample ID per line that will be used for training/testing
    @param outPrefix - the output prefix for final models and other data
    @param splitByType - if True, this will train one model per variant/zygosity type; otherwise a single global model
    @param numProcs - number of processes to use in parallel
    '''
    #we will build a list of numpy arrays, then stack at the end
    tpList = []
    fpList = []

    #get the samples
    samples = parseSlids(slids)

    #load the fields from the first file
    fp = open('%s/%s_tp_fields.json' % (featureDir, samples[0]))
    fieldsList = json.load(fp)
    fp.close()
    
    print(f'[{datetime.datetime.now()}] Full sample list: {samples}')
    #get data from each sample
    for sample in samples:
        #TP first
        tpFN = '%s/%s_tp.npy' % (featureDir, sample)
        tpOrder = '%s/%s_tp_fields.json' % (featureDir, sample)
        fp = open(tpOrder, 'r')
        tpFields = json.load(fp)
        fp.close()
        assert(tpFields == fieldsList)
        tpVar = np.load(tpFN, 'r')
        tpList.append(tpVar)

        #now false positives
        fpFN = '%s/%s_fp.npy' % (featureDir, sample)
        fpOrder = '%s/%s_fp_fields.json' % (featureDir, sample)
        fp = open(fpOrder, 'r')
        fpFields = json.load(fp)
        fp.close()
        assert(fpFields == fieldsList)
        fpVar = np.load(fpFN, 'r')
        fpList.append(fpVar)

    results = {}
    trainedModelDict = {}
    rocDict = {}
    if splitByType:
        #go through each variant/call type pairing and get results for it
        for variantType in [VAR_SNP, VAR_INDEL]:
            for callType in [GT_REF_HET, GT_ALT_HOM, GT_HET_HET]:
                print('[%s] Beginning global, filtered training: %s %s' % (str(datetime.datetime.now()), variantType, callType))
                subResults, subTrainedModelDict, subRocDict = trainAllClassifiers(tpList, fpList, fieldsList, variantType, callType, numProcs)
                subKey = str(variantType)+'_'+str(callType)
                results[subKey] = subResults
                trainedModelDict[subKey] = subTrainedModelDict
                rocDict[subKey] = subRocDict
    else:
        #no filtering
        print('[%s] Beginning global, non-filtered training' % (str(datetime.datetime.now()), ))
        variantType = -1
        callType = -1
        subResults, subTrainedModelDict, subRocDict = trainAllClassifiers(tpList, fpList, fieldsList, variantType, callType, numProcs)
        subKey = 'all_all'
        results[subKey] = subResults
        trainedModelDict[subKey] = subTrainedModelDict
        rocDict[subKey] = subRocDict

    #write outputs
    jsonFN = '%s/stats.json' % (outPrefix, )
    print('[%s] Saving stats to "%s"...' % (str(datetime.datetime.now()), jsonFN))
    fp = open(jsonFN, 'w+')
    json.dump(results, fp, indent=4, sort_keys=True, cls=NumpyEncoder)
    fp.close()

    modelPickleFN = '%s/models.p' % (outPrefix, )
    print('[%s] Saving models to "%s"...' % (str(datetime.datetime.now()), modelPickleFN))
    fp = open(modelPickleFN, 'wb+')
    pickle.dump(trainedModelDict, fp)
    fp.close()

    rocFN = '%s/rocs.json' % (outPrefix, )
    print('[%s] Saving ROC stats to "%s"...' % (str(datetime.datetime.now()), rocFN))
    fp = open(rocFN, 'w+')
    json.dump(rocDict, fp, indent=4, sort_keys=True)
    fp.close()
    
    print('[%s] All models finished training!' % (str(datetime.datetime.now()), ))

def trainAllClassifiers(raw_tpList, raw_fpList, raw_featureLabels, variantType, callType, numProcs):
    '''
    This performs the actual training of the models for us
    @param raw_tpList - a list of numpy arrays corresponding to true positive variants
    @param raw_fpList - a list of numpy arrays corresponding to false positive variants
    @param raw_featureLabels - the ordered feature labels for the tpList and fpList arrays
    @param variantType - the allowed variant type
    @param callType - the allowed call type
    @param numProcs - number of processes to use in parallel
    @return - a dictionary of many results
    '''
    ret = {}
    modelRet = {}
    rocRet = {}

    filterEnabled = (variantType != -1 or callType != -1)
    if filterEnabled:
        assert(variantType != -1 and callType != -1)
    
    configuration = {
        'NUM_PROCESSES' : numProcs,
        'NUM_GROUPS' : len(raw_tpList)
    }
        
    #do all filtering at this stage for ease downstream
    REMOVED_LABELS = []
    if filterEnabled:
        print('[%s] Filtering variants by type: %s %s' % (str(datetime.datetime.now()), variantType, callType))
        flInd = raw_featureLabels.index('VAR-TYPE')
        flInd2 = raw_featureLabels.index('CALL-GT')
        REMOVED_LABELS += ['VAR-TYPE', 'CALL-GT']

    if MANUAL_FS:
        REMOVED_LABELS += MANUAL_FS_LABELS
        print('[%s] Manual feature selection: %s' % (str(datetime.datetime.now()), REMOVED_LABELS))

    if len(REMOVED_LABELS) > 0:
        #get the indices to remove and update features appropriately
        removedIndices = [raw_featureLabels.index(rfl) for rfl in REMOVED_LABELS]
        assert(-1 not in removedIndices)
        featureLabels = [v for i, v in enumerate(raw_featureLabels) if (i not in removedIndices)]

    if not FLIP_TP:
        raise Exception('NO_IMPL for False FLIP_TP')

    train_list_X = []
    train_list_Y = []
    train_groups = []
    test_list_X = []
    test_list_Y = []
    
    #we will need the totals if we auto-calculate the target recalls
    raw_total_tp = 0
    raw_total_fp = 0

    for i in range(0, len(raw_tpList)):
        tpVals = raw_tpList[i]
        fpVals = raw_fpList[i]
        
        if filterEnabled:
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
        
        #store the total number we found before any sort of subsetting; this is used to derive overall accuracies if auto-enabled
        raw_total_tp += tpVals.shape[0]
        raw_total_fp += fpVals.shape[0]

        #we aren't doing a full test, so cut the input sizes down
        if USE_SUBSET:
            tpVals = tpVals[:SUBSET_SIZE]
            fpVals = fpVals[:SUBSET_SIZE]

        #now we need to pull out final train/test sets
        if fpVals.shape[0] == 1:
            '''
            TODO: this is a hack to make stratify work when there is exactly one false positive; 
            the issue is that a class represented once can't be split between the train/test; 
            what is the "correct" way to handle this? 
            - I don't think the below approach of replicating the FP is the correct approach, it's just the one that works here
            - We could maybe remove the FP altogether? also seems bad though
            - maybe it doesn't matter because we really won't use these models due to low N (Complex Het SNV is the failure)
            '''
            combined_X = np.vstack([tpVals, fpVals, fpVals])
            combined_Y = np.array([1]*tpVals.shape[0] + [0, 0])
        else:
            combined_X = np.vstack([tpVals, fpVals])
            combined_Y = np.array([1]*tpVals.shape[0] + [0]*fpVals.shape[0])
        if FLIP_TP:
            combined_Y = 1 - combined_Y
        train_X, test_X, train_Y, test_Y = train_test_split(combined_X, combined_Y, random_state=0, stratify=combined_Y, test_size=TEST_FRACTION)

        #fill out the training/testing arrays and add in groups for the CV training
        train_list_X.append(train_X)
        train_list_Y.append(train_Y)
        train_groups += [i]*train_X.shape[0]
        test_list_X.append(test_X)
        test_list_Y.append(test_Y)

        #these are the filtered versions (assuming some filtering happened above)
        print('', tpVals.shape, fpVals.shape, sep='\t')
    
    #reformat everything into appropriate numpy modes
    final_train_X = np.vstack(train_list_X)
    final_train_Y = np.hstack(train_list_Y)
    final_train_groups = np.array(train_groups)
    final_test_X = np.vstack(test_list_X)
    final_test_Y = np.hstack(test_list_Y)

    print('Final train size:', final_train_X.shape)
    #print(final_train_Y.shape)
    #print(final_train_groups.shape)
    print('Final test size:', final_test_X.shape)
    #print(final_test_Y.shape)

    #TODO: future possible optimization - RandomForest and EasyEnsemble have a "n_jobs" parameters for parallel 
    # fit/predict; is there a way we can utilize that in general?

    print('Raw Total TP, FP:', raw_total_tp, raw_total_fp)
    total_obs = raw_total_fp + raw_total_tp
    raw_precision = raw_total_tp / total_obs
    print('Raw precision:', raw_precision)

    if ENABLE_AUTO_TARGET:
        #we need to derive what the recall targets should be
        missed_precision = GLOBAL_AUTO_TARGET_PRECISION - raw_precision

        if missed_precision < 0:
            #TODO: what should we do here generally? this isn't currently an issue
            raise Exception('raw precision is already better than global target, this is currently unhandled')
        
        total_gap = 1.0 - raw_precision
        lower_target_recall = missed_precision / total_gap
        delta_gap = (1.0 - lower_target_recall) / AUTO_TARGET_BREAKPOINT_COUNT
        
        auto_targets = [lower_target_recall + i*delta_gap for i in range(0, AUTO_TARGET_BREAKPOINT_COUNT)]
        auto_targets += [0.9999, 1.0000]

        #now reverse sort them
        recall_targets = np.array(sorted(auto_targets)[::-1])

    else:
        #manual targetting only
        recall_targets = MANUAL_TARGETS
    print('Recall targets: ', recall_targets)
    
    #go through each model, one at a time
    for (label, raw_clf, hyperparameters, rangeHyperparams) in CLASSIFIERS:
        print('[%s] Starting training for %s...' % (str(datetime.datetime.now()), label))
        
        #this will do training and/or GridSearchCV for us
        if TRAINING_MODE == BAYES_MODE:
            passParams = rangeHyperparams
        else:
            passParams = hyperparameters

        if ENABLE_FEATURE_SELECTION:
            pipeline_clf = Pipeline(
                [
                    #this is a relatively small and relatively shallow GBC
                    ("reduce_dim", 'passthrough'),
                    #now the main classifier
                    ("classifier", raw_clf)
                ]
            )
            new_params = {
                'reduce_dim' : FEATURE_SELECTION_MODELS
            }
            for k in passParams.keys():
                new_key = f'classifier__{k}'
                new_params[new_key] = passParams[k]
            passParams = new_params
        else:
            pipeline_clf = raw_clf 

        fullClf, sumRet, sumRocRet = trainClassifier(
            pipeline_clf, passParams, final_train_X, final_train_Y, final_train_groups, 
            configuration, recall_targets, raw_precision
        )

        if label == 'GradientBoosting':
            if ENABLE_FEATURE_SELECTION:
                print('n_estimators_', fullClf[1].n_estimators_)
            else:
                print('n_estimators_', fullClf.n_estimators_)
        if ENABLE_FEATURE_SELECTION:
            print('support_', fullClf[0].support_)
        #this is the test on the held out test set
        print('[%s]\tFull testing classifier...' % (str(datetime.datetime.now()), ))
        resultsDict = testClassifier(fullClf, final_train_X, final_train_Y, final_test_X, final_test_Y)
        
        #get results and store everything in the dictionary locations below
        allRet, allRocRet = summarizeResults([resultsDict], [final_test_Y], recall_targets, raw_precision)

        ret[label] = {
            'LEAVEONEOUT_SUMMARY' : sumRet,
            'ALL_SUMMARY' : allRet,
            'RAW_TOTAL_TP' : raw_total_tp,
            'RAW_TOTAL_FP' : raw_total_fp
        }
        modelRet[label] = {
            'FEATURES' : featureLabels,
            'FILTER_CALL_TYPE' : (callType if filterEnabled else -1),
            'FILTER_VAR_TYPE' : (variantType if filterEnabled else -1),
            'MODEL' : fullClf,
        }
        rocRet[label] = {
            'LEAVEONEOUT_ROCS' : sumRocRet,
            'ALL_ROC' : allRocRet
        }
        print('[%s] Finished training for %s.' % (str(datetime.datetime.now()), label))
    
    return ret, modelRet, rocRet

def trainClassifier(raw_clf, hyperparameters, train_X, train_Y, train_groups, configuration, recall_targets, base_precision):
    '''
    This will run a classifier for us
    @param raw_clf - a classifier instance
    @param hyperparameters - hyperparameters to use (if enabled)
    @param train_X - the training features
    @param train_Y - the classifier values expected
    @param train_groups - the groups of input values
    @param configuration - a dictionary containing information on how to do the training
    @param recall_targets - the recall targets to evaluate results
    @param raw_precision - the base precision prior to any training
    @return - a trained classifier
    '''
    print('[%s]\tFull training classifier...' % (str(datetime.datetime.now()), ))
    #CONFIGURATION
    NUM_PROCESSES = configuration.get('NUM_PROCESSES', 1)
    NUM_GROUPS = configuration.get('NUM_GROUPS', 1)
    #END-CONFIGURATION

    if TRAINING_MODE == EXACT_MODE:
        print('[%s]\t\tRunning in EXACT_MODE with training only' % (str(datetime.datetime.now()), ))
        clf = raw_clf
        clf.fit(train_X, train_Y)
    elif TRAINING_MODE in [GRID_MODE, BAYES_MODE]:
        cv = LeaveOneGroupOut()
        #scoringMode = 'average_precision' #very little difference, but this one was less consistent
        scoringMode = 'roc_auc'
        if TRAINING_MODE == GRID_MODE:
            print('[%s]\t\tRunning in GRID_MODE with cross-validation, hyperparameter tuning, and training' % (str(datetime.datetime.now()), ))
            gsClf = GridSearchCV(raw_clf, hyperparameters, cv=cv, scoring=scoringMode, n_jobs=NUM_PROCESSES, verbose=1)
        elif TRAINING_MODE == BAYES_MODE:
            print('[%s]\t\tRunning in BAYES_MODE with cross-validation, hyperparameter tuning, and training' % (str(datetime.datetime.now()), ))
            NUM_ITERATIONS = 30
            parallelPoints = max(1, int(np.floor(NUM_PROCESSES / NUM_GROUPS)))
            gsClf = BayesSearchCV(raw_clf, hyperparameters, 
                cv=cv, scoring=scoringMode, 
                n_jobs=NUM_PROCESSES, n_iter=NUM_ITERATIONS, n_points=parallelPoints,
                random_state=0, verbose=1
            )
        else:
            raise Exception('NO_IMPL')
        gsClf.fit(train_X, train_Y, groups=train_groups)
        print('[%s]\t\tBest params: %s' % (str(datetime.datetime.now()), gsClf.best_params_))
        
        print('[%s]\tGathering CV results...'% (str(datetime.datetime.now()), ))
        clf = gsClf.best_estimator_
        results = []
        test_Ys = []
        for train_ind, test_ind in cv.split(train_X, train_Y, train_groups):
            resultsDict = testClassifier(clf, train_X[train_ind], train_Y[train_ind], train_X[test_ind], train_Y[test_ind])
            results.append(resultsDict)
            test_Ys.append(train_Y[test_ind])

        print('[%s]\tGenerating CV summary...'% (str(datetime.datetime.now()), ))
        sumRet, sumRocRet = summarizeResults(results, test_Ys, recall_targets, base_precision)
    else:
        raise Exception('Unexpected mode')
    
    if TRAINING_MODE == EXACT_MODE:
        return clf, [], []
    elif TRAINING_MODE in [GRID_MODE, BAYES_MODE]:
        return clf, sumRet, sumRocRet
    else:
        raise Exception('NO_IMPL')

def testClassifier(clf, train_X, train_Y, test_X, test_Y):
    '''
    This performs the actual training of the models for us
    This will run a classifier for us
    @param clf - a trained classifier instance
    @param train_X - the training features
    @param train_Y - the classifier values expected
    @param test_X - the training features
    @param test_Y - the classifier values expected
    @return - a dictionary of many results
        TEST_PRED - the probability of being in class 1 (i.e. predicted false call) on the test data
        TEST_ROC - tuple (false_positive rate, true_positive_rate, thresholds) from the roc_curve function on the test data
        TEST_ROC_AUC - ROC-AUC from auc(roc_curve(...))
        TRAIN_PRED - the probability of being class 1 on the training data
        TRAIN_ROC - tuple (false_positive rate, true_positive_rate, thresholds) from the roc_curve function on the training data
    '''
    ret = {}
    
    y_pred_rf = clf.predict_proba(test_X)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_Y, y_pred_rf)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    print('[%s]\t\tROC AUC: %f' % (str(datetime.datetime.now()), roc_auc))
    ret['TEST_PRED'] = y_pred_rf
    ret['TEST_ROC'] = (false_positive_rate, true_positive_rate, thresholds)
    ret['TEST_ROC_AUC'] = roc_auc
    
    #binary search for the maximum threshold such that recall is satisfied
    train_y_pred_rf = clf.predict_proba(train_X)[:, 1]
    train_false_positive_rate, train_true_positive_rate, train_thresholds = roc_curve(train_Y, train_y_pred_rf)
    ret['TRAIN_PRED'] = train_y_pred_rf
    ret['TRAIN_ROC'] = (train_false_positive_rate, train_true_positive_rate, train_thresholds)

    return ret

def summarizeResults(results, test_Ys, recallValues, base_precision):
    '''
    This will summarize results for us across multiple leave-one-out runs
    @param results - a list of results dictionaries for a classifier; this is basically a list of results from testClassifier(...)
    @param test_Ys - a list of lists of correct Y-values (e.g. categories)
    @param recallValues - the recall values to evaluate the models at
    @param base_precision - the base precision of the caller without any ML
    @return - tuple (stats, rocs)
        stats - a dict where key is a recall value and value is a dictionary of stats for the model with that recall including test/train TPR, FPR, etc.
        rocs - a list of test ROCs, one for each in results
    '''
    print('[%s]\tRunning threshold tests...' % (str(datetime.datetime.now()), ))
    #TODO: if we want to do confidence intervals on errors, look into it here: (For now, I think CV is fine)
    #https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/
    #note for future matt, if you get 0 errors out of N classifications, we would need to assume the NEXT
    #thing seen is an error and calculate with 1 error out of (N+1) classifications to get our bounds
    #according to math: if we want to have a 99% confidence interval from [99.9%, 100.0%] TPR, then we need
    # at least N=2580 false positive variants (labeled TPR here) with an error <= 1/2580; in other words, if we had
    # 2580 false positive variants and ONLY missed one, we would qualify for the 99.9-100% accuracy range with 
    # 99% confidence in that interval
    
    ret = {}
    if ENABLE_AUTO_TARGET:
        print('', 'tarTPR', 'train_FPR', 'train_TPR', 'test_FPR', 'test_TPR', 'global_prec', 'adjConf', sep='\t')
    else:
        print('', 'tarTPR', 'train_FPR', 'train_TPR', 'test_FPR', 'test_TPR', 'adjConf', sep='\t')
    for minRecall in recallValues:
        #for each specified recall level, we want to calculate how well the training/testing does for a bunch of stats:
        #   training FPR, training TPR, training thresholds, testing FPR, testing TPR, and testing confusion matrix (CM)
        tarTPR = minRecall
        trainFprList = []
        trainTprList = []
        trainThreshList = []

        testCMList = []
        testFprList = []
        testTprList = []

        #for CV, we will have multiple training/testing sets, so we need to gather them all for mean/std later
        #this function is also used for final training, which will only have one set in `results` (i.e. N=1, st. dev.=0)
        for i, resultsDict in enumerate(results):
            #get the training ROC values
            train_false_positive_rate, train_true_positive_rate, train_thresholds = resultsDict['TRAIN_ROC']
            test_Y = test_Ys[i]

            #first, find the point in the training values that matches our recall requirement
            #this will allow us to pick the threshold value matching that recall req.
            ind = bisect.bisect_left(train_true_positive_rate, minRecall)
            while train_true_positive_rate[ind] < minRecall:
                ind += 1
            thresh = train_thresholds[ind]
            
            #now evaluate all test values using that new threshold
            y_pred = resultsDict['TEST_PRED']
            adjPred = [1 if y >= thresh else 0 for y in y_pred]
            adjConf = confusion_matrix(test_Y, adjPred)
            test_FPR = adjConf[0, 1] / np.sum(adjConf[0, :])
            test_TPR = adjConf[1, 1] / np.sum(adjConf[1, :])
            
            #now add values in here
            trainFprList.append(train_false_positive_rate[ind])
            trainTprList.append(train_true_positive_rate[ind])
            trainThreshList.append(thresh)
            testCMList.append(adjConf)
            testFprList.append(test_FPR)
            testTprList.append(test_TPR)

        #dump some output to the screen for sanity
        printVals = [
            ('%s', ''),
            ('%0.4f', minRecall),
            ('%0.4f+-%0.4f', (np.mean(trainFprList), np.std(trainFprList))),
            ('%0.4f+-%0.4f', (np.mean(trainTprList), np.std(trainTprList))),
            ('%0.4f+-%0.4f', (np.mean(testFprList), np.std(testFprList))),
            ('%0.4f+-%0.4f', (np.mean(testTprList), np.std(testTprList)))
        ]
        if ENABLE_AUTO_TARGET:
            recall = np.mean(testTprList)
            printVals += [
                ('%0.6f', base_precision + (1-base_precision) * recall)
            ]
        printVals += [
            ('%s', str(sum(testCMList)).replace('\n', '')),    
        ]
        print('\t'.join([t[0] % t[1] for t in printVals]))

        #save the lists in our output tied to this recall value
        ret[minRecall] = {
            'TRAIN_FPR' : trainFprList,
            'TRAIN_TPR' : trainTprList,
            'TRAIN_THRESHOLD' : trainThreshList,
            'TEST_FPR' : testFprList,
            'TEST_TPR' : testTprList,
            'TEST_CM' : np.array(testCMList).tolist()
        }

    #also pull out all the ROC curves
    retRocs = []
    for resultsDict in results:
        retRocs.append([arr.tolist() for arr in resultsDict['TEST_ROC']])

    return ret, retRocs

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Scripts for training a model to identify variants that should be confirmed"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-p', '--processes', dest='processes', type=int, default=1, help='the number of processes to use (default: 1)')
    p.add_argument('-s', '--split-by-type', dest='split_by_type', action='store_true', default=False, help='split into multiple models by variant/zygosity types (default: False)')

    #required main arguments
    p.add_argument('feature_dir', type=str, help='directory containing extracted features')
    p.add_argument('slids', type=str, help='the list of slids separate by commas (ex: "SL123456-SL123467,SL333333") or a json file containing sample information')
    p.add_argument('output_prefix', type=str, help='prefix to save output files to')

    #parse the arguments
    args = p.parse_args()

    trainModels(args.feature_dir, args.slids, args.output_prefix, args.split_by_type, args.processes)