
import argparse as ap
import eli5
import json
import numpy as np
import pickle

from EvaluateVariants import getClinicalModel
from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE

def gatherEli5Stats(modelDir, minRecall, targetRecall):
    '''
    This will gather our clinical model statistics in an ingestible format
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
    
    #get the stats now and send 'em on back
    return gatherClinicalModelStats(stats, minRecall, targetRecall, models)

def gatherClinicalModelStats(allStats, acceptedRecall, targetRecall, allModels):
    '''
    This will run eli5 if it can and store the results in a dictionary
    @param allStats - the full stats dict (all models)
    @param acceptedRecall - the minimum recall we need
    @param targetRecall - the target recall we want
    @param allModels - the actual loaded models
    @return - a dictionary where keys are the variant/genotype and value has the model name and eli5 outputs
    '''
    ret = {}
    for k in sorted(allStats.keys()):
        stats = allStats[k]
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
        
        #get the clinical model
        clinicalModelDict = getClinicalModel(stats, acceptedRecall, targetRecall)
        bestModelName = clinicalModelDict['model_name']
        
        if bestModelName == None:
            #this is the unfortunate event that NO model passes 
            ret[reformKey] = {
                'best_model' : None
            }
        else:
            #get our classifier and the feature names
            clf = allModels[k][bestModelName]['MODEL']
            coreFeatureNames = [tuple(f.split('-')) for f in allModels[k][bestModelName]['FEATURES']]
            
            #get the weights and add it to the result
            weights = eli5.explain_weights(clf, feature_names=['-'.join(cfn) for cfn in coreFeatureNames])
            dictForm = eli5.formatters.as_dict.format_as_dict(weights)
            ret[reformKey] = {
                'best_model' : bestModelName,
                'eli5' : dictForm
            }

    return ret

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for summarizing feature importances from final models"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-m', '--min-recall', dest='min_recall', default=0.99, type=float, help='the minimum recall for clinical applications (default: 0.990)')
    p.add_argument('-t', '--target-recall', dest='target_recall', default=0.995, type=float, help='the target recall for clinical applications (default: 0.995)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    p.add_argument('out_json', type=str, help='output filename containing eli5 stats (.json)')
    
    #parse the arguments
    args = p.parse_args()

    #gather the stats
    finalStats = gatherEli5Stats(args.model_directory, args.min_recall, args.target_recall)

    fp = open(args.out_json, 'w+')
    json.dump(finalStats, fp, indent=4, sort_keys=True)
    fp.close()
