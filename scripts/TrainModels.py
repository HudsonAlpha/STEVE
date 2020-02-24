
#typical imports
import argparse as ap
import bisect
import datetime
import json
import numpy as np
import pickle

#learning related imports
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

#custom imports
from ExtractFeatures import VAR_SNP, VAR_INDEL, GT_REF_HET, GT_ALT_HOM, GT_HET_HET, GT_REF_HOM
from RunTrainingPipeline import parseSlids

#define the training modes here
EXACT_MODE = 0
GRID_MODE = 1

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
    json.dump(results, fp, indent=4, sort_keys=True)
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

    #parameters we will use for all training
    configuration = {
        'TRAINING_MODE' : GRID_MODE,
        'USE_SUBSET' : False, #if True, restricts input size to "SUBSET_SIZE" for each sample
        'SUBSET_SIZE' : 100000, #this only matter if USE_SUBSET is True
        'FILTER_VARIANTS_BY_TYPE' : filterEnabled,
        'FILTER_VAR_TYPE' : variantType,
        'FILTER_CALL_TYPE' : callType,
        'MANUAL_FS' : True,
        'FLIP_TP' : True,
        'NUM_PROCESSES' : numProcs
    }
    
    FILTER_VARIANTS_BY_TYPE = configuration['FILTER_VARIANTS_BY_TYPE'] #if True, filter down to a particular type of variant (see next two configs)
    FILTER_VAR_TYPE = configuration['FILTER_VAR_TYPE']#set the type of variant to allow through
    FILTER_CALL_TYPE = configuration['FILTER_CALL_TYPE'] #set the call of the variant to allow through
    MANUAL_FS = configuration['MANUAL_FS'] #if True, manually remove some features that are generally useless
    
    #do all filtering at this stage for ease downstream
    REMOVED_LABELS = []
    if FILTER_VARIANTS_BY_TYPE:
        print('[%s] Filtering variants by type: %s %s' % (str(datetime.datetime.now()), FILTER_VAR_TYPE, FILTER_CALL_TYPE))
        flInd = raw_featureLabels.index('VAR-TYPE')
        flInd2 = raw_featureLabels.index('CALL-GT')
        REMOVED_LABELS += ['VAR-TYPE', 'CALL-GT']

    if MANUAL_FS:
        REMOVED_LABELS += ['CALL-ADO', 'CALL-AFO']
        print('[%s] Manual feature selection: %s' % (str(datetime.datetime.now()), REMOVED_LABELS))

    if len(REMOVED_LABELS) > 0:
        #get the indices to remove and update features appropriately
        removedIndices = [raw_featureLabels.index(rfl) for rfl in REMOVED_LABELS]
        assert(-1 not in removedIndices)
        featureLabels = [v for i, v in enumerate(raw_featureLabels) if (i not in removedIndices)]

    FLIP_TP = configuration.get('FLIP_TP', True) #if True, mark false positive as true positives and vice versa
    USE_SUBSET = configuration.get('USE_SUBSET', False) #if True, then only a portion of the data will be tested on (for debugging mainly)
    SUBSET_SIZE = configuration.get('SUBSET_SIZE', 10000) #the size of the subset to use if the previous value is True    
    CURRENT_MODE = configuration.get('TRAINING_MODE', EXACT_MODE) #set the type of analysis we are doing
    
    if not FLIP_TP:
        raise Exception('NO_IMPL for False FLIP_TP')

    train_list_X = []
    train_list_Y = []
    train_groups = []
    test_list_X = []
    test_list_Y = []

    for i in range(0, len(raw_tpList)):
        tpVals = raw_tpList[i]
        fpVals = raw_fpList[i]
        
        if FILTER_VARIANTS_BY_TYPE:
            #figure out which variant match the filter criteria
            tpSearchCrit = (tpVals[:, flInd] == FILTER_VAR_TYPE) & (tpVals[:, flInd2] == FILTER_CALL_TYPE)
            fpSearchCrit = (fpVals[:, flInd] == FILTER_VAR_TYPE) & (fpVals[:, flInd2] == FILTER_CALL_TYPE)
            
            #now extract them and replace them
            tpVals = tpVals[tpSearchCrit, :]
            fpVals = fpVals[fpSearchCrit, :]
        
        #if we have labels to remove, strip them out here
        if len(REMOVED_LABELS) > 0:
            tpVals = np.delete(tpVals, removedIndices, 1)
            fpVals = np.delete(fpVals, removedIndices, 1)
        
        #we aren't doing a full test, so cut the input sizes down
        if USE_SUBSET:
            tpVals = tpVals[:SUBSET_SIZE]
            fpVals = fpVals[:SUBSET_SIZE]

        #now we need to pull out final train/test sets
        TEST_FRACTION = 0.5
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

    #now enumerate the models
    classifiers = [
        ('RandomForest', RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4, n_estimators=200, min_samples_split=2, max_features='sqrt'),
        {
            'random_state' : [0],
            'class_weight' : ['balanced'],
            'n_estimators' : [100, 200],
            'max_depth' : [3, 4],
            'min_samples_split' : [2],
            'max_features' : ['sqrt']
        }),
        #"The most important parameters are base_estimator, n_estimators, and learning_rate" - https://chrisalbon.com/machine_learning/trees_and_forests/adaboost_classifier/
        ('AdaBoost', AdaBoostClassifier(random_state=0, algorithm='SAMME', learning_rate=1.0, base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200),
        {
            'random_state' : [0],
            'base_estimator' : [DecisionTreeClassifier(max_depth=2)],#, SVC(probability=True)],
            'n_estimators' : [100, 200],
            'learning_rate' : [0.01, 0.1, 1.0],
            'algorithm' : ['SAMME', 'SAMME.R']
        }),
        #" Most data scientist see number of trees, tree depth and the learning rate as most crucial parameters" - https://www.datacareer.de/blog/parameter-tuning-in-gradient-boosting-gbm/
        ('GradientBoosting', GradientBoostingClassifier(random_state=0, learning_rate=0.1, loss='exponential', max_depth=4, max_features='sqrt', n_estimators=200),
        {
            'random_state' : [0],
            'n_estimators' : [100, 200],
            'max_depth' : [3, 4],
            'learning_rate' : [0.05, 0.1, 0.2],
            'loss' : ['deviance', 'exponential'],
            'max_features' : ['sqrt'],
            #'subsample' : [0.5, 1.0] #TODO: this could lead to performance increase, should try in future revisions
        }),
        ('EasyEnsemble', EasyEnsembleClassifier(random_state=0, n_estimators=50),
        {
            'random_state' : [0],
            'n_estimators' : [10, 20, 30, 40, 50]
        })
    ]
    
    #go through each model, one at a time
    for (label, raw_clf, hyperparameters) in classifiers:
        print('[%s] Starting training for %s...' % (str(datetime.datetime.now()), label))
        
        #this will do training and/or GridSearchCV for us
        fullClf, sumRet, sumRocRet = trainClassifier(raw_clf, hyperparameters, final_train_X, final_train_Y, final_train_groups, configuration)

        #this is the test on the held out test set
        print('[%s]\tFull testing classifier...' % (str(datetime.datetime.now()), ))
        resultsDict = testClassifier(fullClf, final_train_X, final_train_Y, final_test_X, final_test_Y)
        
        #get results and store everything in the dictionary locations below
        allRet, allRocRet = summarizeResults([resultsDict], [final_test_Y])

        ret[label] = {
            'LEAVEONEOUT_SUMMARY' : sumRet,
            'ALL_SUMMARY' : allRet
        }
        modelRet[label] = {
            'FEATURES' : featureLabels,
            'FILTER_CALL_TYPE' : (FILTER_CALL_TYPE if FILTER_VARIANTS_BY_TYPE else -1),
            'FILTER_VAR_TYPE' : (FILTER_VAR_TYPE if FILTER_VARIANTS_BY_TYPE else -1),
            'MODEL' : fullClf,
        }
        rocRet[label] = {
            'LEAVEONEOUT_ROCS' : sumRocRet,
            'ALL_ROC' : allRocRet
        }
        print('[%s] Finished training for %s.' % (str(datetime.datetime.now()), label))
    
    return ret, modelRet, rocRet

def trainClassifier(raw_clf, hyperparameters, train_X, train_Y, train_groups, configuration):
    '''
    This will run a classifier for us
    @param raw_clf - a classifier instance
    @param hyperparameters - hyperparameters to use (if enabled)
    @param train_X - the training features
    @param train_Y - the classifier values expected
    @param train_groups - the groups of input values
    @param configuration - a dictionary containing information on how to do the training
    @return - a trained classifier
    '''
    print('[%s]\tFull training classifier...' % (str(datetime.datetime.now()), ))
    #CONFIGURATION
    CURRENT_MODE = configuration.get('TRAINING_MODE', EXACT_MODE) #set the type of analysis we are doing
    NUM_PROCESSES = configuration.get('NUM_PROCESSES', 1)
    #END-CONFIGURATION

    if CURRENT_MODE == EXACT_MODE:
        print('[%s]\t\tRunning in EXACT_MODE with training only' % (str(datetime.datetime.now()), ))
        clf = raw_clf
        clf.fit(train_X, train_Y)
    elif CURRENT_MODE == GRID_MODE:
        print('[%s]\t\tRunning in GRID_MODE with cross-validation, hyperparameter tuning, and training' % (str(datetime.datetime.now()), ))
        cv = LeaveOneGroupOut()
        scoringMode = 'roc_auc'
        #scoringMode = 'average_precision' #very little difference, but this one was less consistent
        gsClf = GridSearchCV(raw_clf, hyperparameters, cv=cv, scoring=scoringMode, n_jobs=NUM_PROCESSES, verbose=1)
        gsClf.fit(train_X, train_Y, train_groups)
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
        sumRet, sumRocRet = summarizeResults(results, test_Ys)
    else:
        raise Exception('Unexpected mode')
    
    if CURRENT_MODE == EXACT_MODE:
        return clf, [], []
    elif CURRENT_MODE == GRID_MODE:
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

def summarizeResults(results, test_Ys):
    '''
    This will summarize results for us across multiple leave-one-out runs
    @param results - a list of results dictionaries for a classifier
    @return - TODO
    '''
    print('[%s]\tRunning threshold tests...' % (str(datetime.datetime.now()), ))
    recallValues = np.array([1.0, 0.9999, 0.999, 0.998, 0.997, 0.996, 0.995, 0.99])
    
    #TODO: if we want to do confidence intervals on errors, look into it here: (For now, I think CV is fine)
    #https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/
    #note for future matt, if you get 0 errors out of N classifications, we would need to assume the NEXT
    #thing seen is an error and calculate with 1 error out of (N+1) classifications to get our bounds
    #according to math: if we want to have a 99% confidence interval from [99.9%, 100.0%] TPR, then we need
    # at least N=2580 false positive variants (labeled TPR here) with an error <= 1/2580; in other words, if we had
    # 2580 false positive variants and ONLY missed one, we would qualify for the 99.9-100% accuracy range with 
    # 99% confidence in that interval
    
    ret = {}
    print('', 'tarTPR', 'train_FPR', 'train_TPR', 'test_FPR', 'test_TPR', 'adjConf', sep='\t')
    for minRecall in recallValues:
        tarTPR = minRecall
        trainFprList = []
        trainTprList = []
        trainThreshList = []

        testCMList = []
        testFprList = []
        testTprList = []
        
        for i, resultsDict in enumerate(results):
            train_false_positive_rate, train_true_positive_rate, train_thresholds = resultsDict['TRAIN_ROC']
            test_Y = test_Ys[i]

            #first, find the point in the training values that matches our recall requirement
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
            ('%0.4f+-%0.4f', (np.mean(testTprList), np.std(testTprList))),
            ('%s', str(sum(testCMList)).replace('\n', '')),    
        ]
        print('\t'.join([t[0] % t[1] for t in printVals]))

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
    p.add_argument('slids', type=str, help='the list of slids separate by commas (ex: "SL123456-SL123467,SL333333")')
    p.add_argument('output_prefix', type=str, help='prefix to save output files to')

    #parse the arguments
    args = p.parse_args()

    trainModels(args.feature_dir, args.slids, args.output_prefix, args.split_by_type, args.processes)