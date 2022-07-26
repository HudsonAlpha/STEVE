
import argparse as ap
import json
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from EvaluateVariants import getClinicalModel
from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE

def printAllStats(modelDir, rocDir, minRecall, targetRecall, global_precision):
    '''
    This will print out our model statistics in an ingestible format
    @param modelDir - the model directory 
    @param rocDir - a directory for output ROC curve images
    @param minRecall - the minimum recall values allowed
    @param targetRecall - the target recall we want the models to achieve
    @param global_precision - float value indicating target global precision; if set, 
        then it will dynamically figure out the target recalls based on the data
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
    
    imageDict = printClinicalModels(stats, minRecall, targetRecall, models, global_precision)
    if rocDir != None:
        createTrainingImage(imageDict, rocDir)

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
        header.append(f'{e}-recall')
        header.append('CV_recall')
        header.append('FPR')
        header.append('CV_FPR')
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

def createTrainingImage(imageDict, rocDir):
    outFN = '%s/clinical_compare.png' % (rocDir, )
    plt.figure()
    plt.title('Selected Clinical Models')
    plt.xlabel('FPR')
    plt.ylabel('TPR / sensitivity')
    plt.xlim([0.0000, 1.0000])
    plt.ylim([0.9900, 1.0000])
    plt.axhline(0.9950, color='black', linestyle='--')

    markers = ['o', 's', 'D', 'v', 'P', '^']
    for i, k in enumerate(imageDict.keys()):
        tprArray, fprArray, tpr, fpr = imageDict[k]
        plt.scatter(fprArray, tprArray, marker=markers[i], facecolors='none', edgecolors='black')
        plt.scatter([fpr], [tpr], marker=markers[i], facecolors='black', edgecolors='black', label=k)

    plt.legend()
    plt.grid()
    plt.savefig(outFN)
    plt.close()

def printClinicalModels(allStats, acceptedRecall, targetRecall, allModels, global_precision):
    header = [
        'variant_type', 'best_model', 'model_eval',
        'CV_recall', 'final_recall', 'CV_FPR', 'final_FPR'
    ]
    if global_precision == None:
        print('[clinical_model min=%0.4f tar=%0.4f]' % (acceptedRecall, targetRecall))
    else:
        print('[clinical_model target_global_precision=%0.4f]' % (global_precision, ))
        header += ['target_recall', 'global_prec']
    print(*header, sep='\t')
    imageDict = {}
    for k in sorted(allStats.keys()):
        stats = allStats[k]
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]

        clinDict = getClinicalModel(stats, acceptedRecall, targetRecall, global_precision)
        bestModelName = clinDict['model_name']

        if bestModelName == None:
            #this is the unfortunate event that NO model passes 
            rowVals = [
                reformKey, 'None', 'None', '--', '--', '--', '--'
            ]
            if global_precision:
                rowVals += ['--', '--']
            print(*rowVals, sep='\t')
        else:
            bestModelTargetRecall = clinDict['eval_recall']
            str_bestModelTargetRecall = '{0:0.4f}'.format(float(bestModelTargetRecall))

            #copy TPR results
            bestTPR = stats[bestModelName]['ALL_SUMMARY'][bestModelTargetRecall]['TEST_TPR'][0]
            bestTPRAvg = np.mean(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_TPR'])
            bestTPRStd = np.std(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_TPR'])
            
            #get FPR results
            bestFPRAvg = np.mean(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_FPR'])
            bestFPRStd = np.std(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_FPR'])
            bestFPR = stats[bestModelName]['ALL_SUMMARY'][bestModelTargetRecall]['TEST_FPR'][0]

            rowVals = [
                reformKey, bestModelName, str_bestModelTargetRecall,
                '%0.4f+-%0.4f' % (bestTPRAvg, bestTPRStd), 
                '%0.4f' % bestTPR,
                '%0.4f+-%0.4f' % (bestFPRAvg, bestFPRStd), 
                '%0.4f' % bestFPR,
                #str(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_TPR']),
                #str(stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_FPR'])
            ]
            if global_precision:
                base_precision = clinDict['base_precision']
                derived_recall = clinDict['derived_recall']
                calculated_precision = base_precision + (1 - base_precision) * bestTPR
                rowVals += [
                    '%0.4f' % (derived_recall, ),
                    '%0.6f' % (calculated_precision, )
                ]

            print(*rowVals, sep='\t')

            imageDict[reformKey] = (
                stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_TPR'],
                stats[bestModelName]['LEAVEONEOUT_SUMMARY'][bestModelTargetRecall]['TEST_FPR'],
                bestTPR,
                bestFPR
            )
    
    return imageDict

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for summarizing training performance"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')
    p.add_argument('-r', '--roc-dir', dest='roc_dir', default=None, help='a directory to store ROC images (default: None)')
    p.add_argument('-m', '--min-recall', dest='min_recall', default=0.99, type=float, help='the minimum recall for clinical applications (default: 0.990)')
    p.add_argument('-t', '--target-recall', dest='target_recall', default=0.995, type=float, help='the target recall for clinical applications (default: 0.995)')
    p.add_argument('-g', '--global-precision', dest='global_precision', default=None, type=float, help='the global precision target; if set, override min/target recalls (default: None)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    
    #parse the arguments
    args = p.parse_args()

    printAllStats(args.model_directory, args.roc_dir, args.min_recall, args.target_recall, args.global_precision)