
import argparse as ap
import json
import numpy as np

from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE

def printAllStats(modelDir):
    '''
    This will print out our model statistics in an ingestible format
    @param modelDir - the model directory 
    '''
    #read in the stats
    jsonFN = '%s/stats.json' % (modelDir, )
    fp = open(jsonFN, 'r')
    stats = json.load(fp)
    fp.close()

    #print the stats for each sub-model
    for k in sorted(stats.keys()):
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
        printModelStats(reformKey, stats[k])

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
    print(modelType, 'TP=%d' % testTN, 'FP=%d' % testTP)
    #print(testResults, np.sum(testResults, axis=1))
    header = [
        'target_recall'
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

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for evaluating variants for false positive status"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    
    #parse the arguments
    args = p.parse_args()

    printAllStats(args.model_directory)