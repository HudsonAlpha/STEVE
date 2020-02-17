
import argparse as ap
import json
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE

def printAllStats(modelDir, rocDir, minRecall, targetRecall):
    '''
    This will print out our model statistics in an ingestible format
    @param modelDir - the model directory 
    @param rocDir - a directory for output ROC curve images
    @param minRecall - the minimum recall values allowed
    @param targetRecall - the target recall we want the models to achieve
    '''
    #read in the stats
    jsonFN = '%s/stats.json' % (modelDir, )
    fp = open(jsonFN, 'r')
    stats = json.load(fp)
    fp.close()

    #read in the models
    modelPickleFN = '%s/models.p' % (modelDir, )
    fp = open(modelPickleFN, 'rb')
    models = pickle.load(fp)
    fp.close()

    if rocDir != None:
        if not os.path.exists(rocDir):
            os.makedirs(rocDir)

        rocFN = '%s/rocs.json' % modelDir
        fp = open(rocFN, 'r')
        rocs = json.load(fp)
        fp.close()

    #print the stats for each sub-model
    for k in sorted(stats.keys()):
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
        printModelStats(reformKey, stats[k])

        if rocDir != None:
            createRocImage(reformKey, rocs[k], rocDir)
    
    printClinicalModels(stats, minRecall, targetRecall, models)

def printModelStats(modelType, stats):
    '''
    This will print out our model statistics for the specific model type
    @param modelType - name of the model training set
    @param stats - the stats for that model
    '''
    availableRecalls = list(stats[list(stats.keys())[0]]['ALL_SUMMARY'].keys())
    evalList = sorted(stats.keys())
    
    #pull out test size also
    testResults = stats[evalList[0]]['ALL_SUMMARY'][availableRecalls[0]]['TEST_CM'][0]
    testTN, testTP = np.sum(testResults, axis=1)
    #print(modelType, 'TP=%d' % testTN, 'FP=%d' % testTP)
    print('[%s TP=%d FP=%d]' % (modelType, testTN, testTP))
    #print(testResults, np.sum(testResults, axis=1))
    header = [
        'eval_recall'
    ]
    for e in evalList:
        header.append(e)
        header.append('')
        header.append('')
        header.append('')
    print(*header, sep='\t')
    for targetRecall in availableRecalls:
        rowVals = ['%0.4f' % (float(targetRecall), )]
        for mn in evalList:
            #rowVals.append('%0.4f (%0.4f)' % (stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_TPR'][0], stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_FPR'][0]))
            #print(stats[mn]['LEAVEONEOUT_SUMMARY'][targetRecall]['TEST_TPR'])
            recallMean = np.mean(stats[mn]['LEAVEONEOUT_SUMMARY'][targetRecall]['TEST_TPR'])
            recallStd = np.std(stats[mn]['LEAVEONEOUT_SUMMARY'][targetRecall]['TEST_TPR'])
            rowVals.append('%0.4f' % (stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_TPR'][0], ))
            rowVals.append('%0.4f+-%0.4f' % (recallMean, recallStd))

            fprMean = np.mean(stats[mn]['LEAVEONEOUT_SUMMARY'][targetRecall]['TEST_FPR'])
            fprStd = np.std(stats[mn]['LEAVEONEOUT_SUMMARY'][targetRecall]['TEST_FPR'])
            rowVals.append('%0.4f' % (stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_FPR'][0], ))
            rowVals.append('%0.4f+-%0.4f' % (fprMean, fprStd))
        print(*rowVals, sep='\t')
    print()

def createRocImage(modelType, rocStats, rocDir):
    '''
    This will create a ROC curve image for the provided data
    @param modelType - name of the model training set
    @param rocStats - the ROC stats for that model
    @param rocDir - the directory to save the image to
    '''
    evalList = sorted(rocStats.keys())
    
    outFN = '%s/%s.png' % (rocDir, modelType)
    plt.figure()

    for mn in evalList:
        fpr, tpr, thresh = rocStats[mn]['ALL_ROC'][0]
        plt.plot(fpr, tpr, label=mn)

    plt.title(modelType)
    plt.xlabel('FPR')
    plt.ylabel('TPR / sensitivity')
    plt.xlim([0.0000, 1.0000])
    plt.ylim([0.9900, 1.0000])
    plt.axhline(0.9950, color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.savefig(outFN)
    plt.close()
    return outFN

def printClinicalModels(allStats, acceptedRecall, targetRecall, allModels):
    header = [
        'variant_type', 'best_model', 'model_eval',
        'CV_recall', 'final_recall', 'CV_FPR', 'final_FPR'
    ]
    print('[clinical_model min=%0.4f tar=%0.4f]' % (acceptedRecall, targetRecall))
    print(*header, sep='\t')
    for k in sorted(allStats.keys()):
        stats = allStats[k]
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]

        bestModelTargetRecall = None
        bestModelName = None
        bestHM = 0.0
        for mn in stats.keys():
            for tr in stats[mn]['ALL_SUMMARY']:
                #CM = confusion matrix
                modelCM = np.array(stats[mn]['ALL_SUMMARY'][tr]['TEST_CM'][0])
                if (np.sum(modelCM[:, 1]) == 0.0 or np.sum(modelCM[1, :]) == 0):
                    modelHM = 0.0
                else:
                    modelRecall = modelCM[1, 1] / (modelCM[1, 0] + modelCM[1, 1])
                    trainTPR = np.array(stats[mn]['LEAVEONEOUT_SUMMARY'][tr]['TEST_TPR'])
                    trainAvg = np.mean(trainTPR)
                    trainStd = np.std(trainTPR)

                    #if the average training low end is too low OR the final model is outside the training bounds
                    # THEN we will not use the model
                    twoSDBottom = trainAvg - 2*trainStd
                    if (twoSDBottom < acceptedRecall or
                        modelRecall < twoSDBottom):
                        modelHM = 0.0
                    else:
                        #in clinical, best is harmonic mean of our adjusted recall and our TNR
                        modelTNR = modelCM[0, 0] / (modelCM[0, 0] + modelCM[0, 1])
                        #adjRecall = (modelRecall*100 - 99)
                        #adjRecall = modelRecall
                        adjRecall = (modelRecall - acceptedRecall) / (float(targetRecall) - acceptedRecall)
                        if adjRecall > 1.0:
                            adjRecall = 1.0
                        modelHM = 2 * adjRecall * modelTNR / (adjRecall+modelTNR)
                        
                if modelHM > bestHM:
                    bestModelTargetRecall = tr
                    bestModelName = mn
                    bestHM = modelHM

                    bestTPRAvg = trainAvg
                    bestTPRStd = trainStd
                    bestTPR = modelRecall

                    bestFPRAvg = np.mean(stats[mn]['LEAVEONEOUT_SUMMARY'][tr]['TEST_FPR'])
                    bestFPRStd = np.std(stats[mn]['LEAVEONEOUT_SUMMARY'][tr]['TEST_FPR'])
                    bestFPR = stats[mn]['ALL_SUMMARY'][tr]['TEST_FPR'][0]
        
        if bestModelName == None:
            #this is the unfortunate event that NO model passes 
            print(reformKey, 'None', 'None', '--', '--', '--', '--', sep='\t')
        else:
            clf = allModels[k][bestModelName]['MODEL']
            coreFeatureNames = [tuple(f.split('-')) for f in allModels[k][bestModelName]['FEATURES']]
            
            modelTargetRecall = bestModelTargetRecall
            evalList = [bestModelName]
            rowVals = [
                reformKey, bestModelName, bestModelTargetRecall,
                '%0.4f+-%0.4f' % (bestTPRAvg, bestTPRStd), 
                '%0.4f' % bestTPR,
                '%0.4f+-%0.4f' % (bestFPRAvg, bestFPRStd), 
                '%0.4f' % bestFPR
            ]
            print(*rowVals, sep='\t')
            #TODO: if you want to explore this, need to import the eli5 (costly, so should make an option)
            #import eli5
            #print(eli5.formatters.text.format_as_text(eli5.explain_weights(clf, feature_names=[str(cfn) for cfn in coreFeatureNames])))
    

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for summarizing training performance"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')
    p.add_argument('-r', '--roc-dir', dest='roc_dir', default=None, help='a directory to store ROC images (default: None)')
    p.add_argument('-m', '--min-recall', dest='min_recall', default=0.99, type=float, help='the minimum recall for clinical applications (default: 0.990)')
    p.add_argument('-t', '--target-recall', dest='target_recall', default=0.995, type=float, help='the target recall for clinical applications (default: 0.995)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    
    #parse the arguments
    args = p.parse_args()

    printAllStats(args.model_directory, args.roc_dir, args.min_recall, args.target_recall)