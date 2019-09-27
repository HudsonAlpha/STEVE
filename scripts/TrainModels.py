
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

#custom imports
from ExtractFeatures import VAR_SNP, VAR_INDEL, GT_REF_HET, GT_ALT_HOM, GT_HET_HET, GT_REF_HOM
from RunTrainingPipeline import parseSlids

#define the training mode here
EXACT_MODE = 0
GRID_MODE = 1

def trainModels(featureDir, slids, outPrefix, splitByType):
    '''
    Trains models given a selection of features
    @param featureDir - the directory containing features
    @param slids - a file with one sample ID per line that will be used for training/testing
    @param outPrefix - the output prefix for final models and other data
    @param splitByType - if True, this will train one model per variant/zygosity type; otherwise a single global model
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
        
    #allTP = np.vstack(tpList)
    #allFP = np.vstack(fpList)
    results = {}
    trainedModelDict = {}
    rocDict = {}
    if splitByType:
        for variantType in [VAR_SNP, VAR_INDEL]:
            for callType in [GT_REF_HET, GT_ALT_HOM, GT_HET_HET]:
                print('[%s] Beginning global, filtered training: %s %s' % (str(datetime.datetime.now()), variantType, callType))
                subResults, subTrainedModelDict, subRocDict = trainAllClassifiers(tpList, fpList, fieldsList, variantType, callType)
                subKey = str(variantType)+'_'+str(callType)
                results[subKey] = subResults
                trainedModelDict[subKey] = subTrainedModelDict
                rocDict[subKey] = subRocDict
    else:
        #no filtering
        print('[%s] Beginning global, non-filtered training' % (str(datetime.datetime.now()), ))
        variantType = -1
        callType = -1
        subResults, subTrainedModelDict, subRocDict = trainAllClassifiers(tpList, fpList, fieldsList, variantType, callType)
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

def trainAllClassifiers(raw_tpList, raw_fpList, raw_featureLabels, variantType, callType):
    '''
    This performs the actual training of the models for us
    @param raw_tpList - a list of numpy arrays corresponding to true positive variants
    @param raw_fpList - a list of numpy arrays corresponding to false positive variants
    @param raw_featureLabels - the ordered feature labels for the tpList and fpList arrays
    @param variantType - the allowed variant type
    @param callType - the allowed call type
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
        'TRAINING_MODE' : EXACT_MODE,
        'USE_SUBSET' : False, #change to false when debugging is complete
        'SUBSET_SIZE' : 1000,
        'FILTER_VARIANTS_BY_TYPE' : filterEnabled,
        'FILTER_VAR_TYPE' : variantType,
        'FILTER_CALL_TYPE' : callType,
        'MANUAL_FS' : True,
        'FLIP_TP' : True
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

    tpList = []
    fpList = []
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
        
        if len(REMOVED_LABELS) > 0:
            tpVals = np.delete(tpVals, removedIndices, 1)
            fpVals = np.delete(fpVals, removedIndices, 1)
            
        #these are the filtered versions (assuming some filtering happened above)
        tpList.append(tpVals)
        fpList.append(fpVals)
        print('', tpVals.shape, fpVals.shape, sep='\t')
    
    #now enumerate the models
    classifiers = [
        #Best params: {'class_weight': 'balanced', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 200, 'random_state': 0}
        ('RandomForest', RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4, n_estimators=200, min_samples_split=2, max_features='sqrt'),
        {
            'random_state' : [0],
            'class_weight' : ['balanced'],
            'n_estimators' : [100, 200],
            'max_depth' : [2, 3, 4],
            'min_samples_split' : [2],
            'max_features' : ['sqrt']
        }),
        #Best params: {'algorithm': 'SAMME', 'base_estimator': DecisionTreeClassifier(max_depth=2,...), 'n_estimators': 150, 'random_state': 0}
        ('AdaBoost', AdaBoostClassifier(random_state=0, algorithm='SAMME', base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=150),
        {
            'random_state' : [0],
            'base_estimator' : [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
            'n_estimators' : [50, 100, 150, 200],
            'algorithm' : ['SAMME', 'SAMME.R']
        }),
        #Best params: {'loss': 'exponential', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200, 'random_state': 0}
        ('GradientBoosting', GradientBoostingClassifier(random_state=0, loss='exponential', max_depth=4, max_features='sqrt', n_estimators=200),
        {
            'random_state' : [0],
            'loss' : ['deviance', 'exponential'],
            'n_estimators' : [100, 200],
            'max_depth' : [3, 4],
            'max_features' : ['sqrt']
        }),
        #Best params: {'n_estimators': 50, 'random_state': 0}
        ('EasyEnsemble', EasyEnsembleClassifier(random_state=0, n_estimators=50),
        {
            'random_state' : [0],
            'n_estimators' : [10, 20, 30, 40, 50]
        })
    ]

    FLIP_TP = configuration.get('FLIP_TP', True) #if True, mark false positive as true positives and vice versa
    USE_SUBSET = configuration.get('USE_SUBSET', False) #if True, then only a portion of the data will be tested on (for debugging mainly)
    SUBSET_SIZE = configuration.get('SUBSET_SIZE', 10000) #the size of the subset to use if the previous value is True    
    
    if not FLIP_TP:
        raise Exception('NO_IMPL for False FLIP_TP')

    #go through each model, one at a time
    for (label, raw_clf, hyperparameters) in classifiers:
        print('[%s] Starting training for %s...' % (str(datetime.datetime.now()), label))
        
        #for each model, we need to do a leave-one-out cross validation based on the samples available
        results = []
        test_Ys = []
        for i in range(0, len(tpList)):
            #"i" is the one being left out
            print('[%s]\tStarting iteration %s' % (str(datetime.datetime.now()), i))

            trainXList = []
            trainYList = []
            for j in range(0, len(tpList)):
                if i == j:
                    #this is the left-out test set for this iteration
                    if USE_SUBSET:
                        test_X = np.vstack([tpList[i][:SUBSET_SIZE], fpList[i][:SUBSET_SIZE]])
                        test_Y = np.array([1]*tpList[i][:SUBSET_SIZE].shape[0] + [0]*fpList[i][:SUBSET_SIZE].shape[0])
                    else:
                        test_X = np.vstack([tpList[i], fpList[i]])
                        test_Y = np.array([1]*tpList[i].shape[0] + [0]*fpList[i].shape[0])
                else:
                    #this is part of the training set
                    if USE_SUBSET:
                        trainXList.append(tpList[j][:SUBSET_SIZE])
                        trainXList.append(fpList[j][:SUBSET_SIZE])
                        trainYList += [1]*tpList[j][:SUBSET_SIZE].shape[0] + [0]*fpList[j][:SUBSET_SIZE].shape[0]
                    else:
                        trainXList.append(tpList[j])
                        trainXList.append(fpList[j])
                        trainYList += [1]*tpList[j].shape[0] + [0]*fpList[j].shape[0]

            #stack the training arrays and shuffle them
            train_X = np.vstack(trainXList)
            train_Y = np.array(trainYList)
            train_X, train_Y = shuffle(train_X, train_Y, random_state=0)

            if FLIP_TP:
                test_Y = 1-test_Y
                train_Y = 1-train_Y

            print('[%s]\tTraining size: %s %s' % (str(datetime.datetime.now()), train_X.shape, train_Y.shape))
            print('[%s]\tTesting size: %s %s' % (str(datetime.datetime.now()), test_X.shape, test_Y.shape))
            print('[%s]\tTraining classifier...' % (str(datetime.datetime.now()), ))
            trained_clf = trainClassifier(raw_clf, hyperparameters, train_X, train_Y, configuration)
            print('[%s]\tTesting classifier...' % (str(datetime.datetime.now()), ))
            resultsDict = testClassifier(trained_clf, train_X, train_Y, test_X, test_Y, featureLabels)
            
            #need these for summary analysis later
            results.append(resultsDict)
            test_Ys.append(test_Y)

        #these is for the leave-one-out CV
        sumRet, sumRocRet = summarizeResults(results, test_Ys)

        #now do the full training
        allTP = np.vstack(tpList)
        allFP = np.vstack(fpList)
        if USE_SUBSET:
            allXs = np.vstack([allTP[:SUBSET_SIZE], allFP[:SUBSET_SIZE]])
            allYs = np.array([1]*allTP[:SUBSET_SIZE].shape[0] + [0]*allFP[:SUBSET_SIZE].shape[0])
        else:
            allXs = np.vstack([allTP, allFP])
            allYs = np.array([1]*allTP.shape[0] + [0]*allFP.shape[0])

        if FLIP_TP:
            allYs = 1 - allYs
        
        #now split this for a quick evaluation afterwards, train on the bulk of the data IMO
        print('[%s]\tFull train_test_split(...)' % (str(datetime.datetime.now()), ))
        train_X, test_X, train_Y, test_Y = train_test_split(allXs, allYs, random_state=0, stratify=allYs, test_size=0.1)
        print('[%s]\tFull training size: %s %s' % (str(datetime.datetime.now()), train_X.shape, train_Y.shape))
        print('[%s]\tFull testing size: %s %s' % (str(datetime.datetime.now()), test_X.shape, test_Y.shape))
        fullClf = trainClassifier(raw_clf, hyperparameters, train_X, train_Y, configuration)

        print('[%s]\tFull testing classifier...' % (str(datetime.datetime.now()), ))
        resultsDict = testClassifier(fullClf, train_X, train_Y, test_X, test_Y, featureLabels)
        
        allRet, allRocRet = summarizeResults([resultsDict], [test_Y])

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

def trainClassifier(raw_clf, hyperparameters, train_X, train_Y, configuration):
    '''
    This will run a classifier for us
    @param raw_clf - a classifier instance
    @param hyperparameters - hyperparameters to use (if enabled)
    @param train_X - the training features
    @param train_Y - the classifier values expected
    @param configuration - a dictionary containing information on how to do the training
    @return - a trained classifier
    '''
    #CONFIGURATION
    CURRENT_MODE = configuration.get('TRAINING_MODE', EXACT_MODE) #set the type of analysis we are doing
    #END-CONFIGURATION

    if CURRENT_MODE == EXACT_MODE:
        clf = raw_clf
    elif CURRENT_MODE == GRID_MODE:
        raise Exception('Not tested yet, verify IMPL first')
        cv = StratifiedKFold(n_splits=10)
        scoringMode = 'roc_auc'
        clf = GridSearchCV(raw_clf, hyperparameters, cv=cv, scoring=scoringMode, n_jobs=16, verbose=1)
    else:
        raise Exception('Unexpected mode')
    
    clf.fit(train_X, train_Y)

    #if CURRENT_MODE in [GRID_MODE]:
    #    print('\tBest params:', clf.best_params_)
    
    return clf

def testClassifier(clf, train_X, train_Y, test_X, test_Y, featureLabels):
    '''
    This performs the actual training of the models for us
    This will run a classifier for us
    @param clf - a trained classifier instance
    @param train_X - the training features
    @param train_Y - the classifier values expected
    @param test_X - the training features
    @param test_Y - the classifier values expected
    @param featureLabels - the ordered feature labels for the tpList and fpList arrays
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
    recallValues = np.array([1.0, 0.9999, 0.999, 0.995, 0.99])
    
    #TODO: if we want to do confidence intervals on errors, look into it here:
    #https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/
    #note for future matt, if you get 0 errors out of N classifications, we would need to assume the NEXT
    #thing seen is an error and calculate with 1 error out of (N+1) classifications to get our bounds
    #according to math: if we want to have a 99% confidence interval from [99.9%, 100.0%] TPR, then we need
    # at least N=2580 false positive variants (labeled TPR here) with an error <= 1/2580; in other words, if we had
    # 2580 false positive variants and ONLY missed one, we would qualify for the 99.9-100% accuracy range with 
    # 99% confidence in that interval
    
    #print('', 'tarTPR', 'ind', 'FPR', 'TPR', 'threshold', 'adjConf', 'test_FPR', 'test_TPR', sep='\t')
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
            
            '''
            #TODO: does this confidence interval stuff factor in still?
            #95% confidence interval uses 1.96 as the constant (see https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)
            if test_TPR == 1.0:
                test_err = 1 / (np.sum(adjConf[1, :])+1)
            else:
                test_err = 1-test_TPR
            confInterval = 1.96*np.sqrt(test_err*(1-test_err)/np.sum(adjConf[1, :]))
            
            #print('\tadj-%.4f' % (t, ), adjConf)
            #print('', minRecall, ind, train_false_positive_rate[ind], train_true_positive_rate[ind], t, *adjConf, test_FPR, '%0.4f+-%0.4f' % (test_TPR, confInterval), sep='\t')
            #print('\t%0.4f\t%d')
            printVals = [
                ('%s', ''),
                ('%0.4f', minRecall),
                ('%d', ind),
                ('%0.4f', train_false_positive_rate[ind]),
                ('%0.4f', train_true_positive_rate[ind]),
                ('%0.4f', t),
                ('%s', str(adjConf).replace('\n', '')),
                ('%0.4f', test_FPR),
                ('%0.4f+-%0.4f', (test_TPR, confInterval))
            ]
            print('\t'.join([t[0] % t[1] for t in printVals]))
            '''

        #print('', 'tarTPR', 'train_FPR', 'train_TPR', 'adjConf', 'test_FPR', 'test_TPR', sep='\t')
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
    #p.add_argument('-d', '--date-subdir', dest='date_subdir', default=None, help='the date subdirectory (default: "hli-YYMMDD")')
    #p.add_argument('-p', '--processes', dest='processes', type=int, default=1, help='the number of processes to use (default: 1)')
    p.add_argument('-s', '--split-by-type', dest='split_by_type', action='store_true', default=False, help='split into multiple models by variant/zygosity types (default: False)')

    #required main arguments
    p.add_argument('feature_dir', type=str, help='directory containing extracted features')
    p.add_argument('slids', type=str, help='the list of slids separate by commas (ex: "SL123456-SL123467,SL333333")')
    p.add_argument('output_prefix', type=str, help='prefix to save output files to')

    #parse the arguments
    args = p.parse_args()

    trainModels(args.feature_dir, args.slids, args.output_prefix, args.split_by_type)