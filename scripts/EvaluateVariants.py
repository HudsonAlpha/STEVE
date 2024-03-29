
import argparse as ap
import cyvcf2
import csv
import json
import numpy as np
import pickle
import re
#import vcf

from ExtractFeatures import ALL_METRICS, getVariantFeatures, GT_TRANSLATE, VAR_TRANSLATE
from TrainingConfig import ENABLE_AUTO_TARGET, GLOBAL_AUTO_TARGET_PRECISION

def getClinicalModel(stats, acceptedRecall, targetRecall, global_precision):
    '''
    @param stats - the stats dictionary from training
    @param acceptedRecall - the minimum acceptable recall
    @param targetRecall - the target we are aiming for, must be greater than or equal to accepted
    @param global_precision - float value indicating target global precision; if set, 
        then it will dynamically figure out the target recalls based on the data
    @return - a dictionary with the eval recall, model name, and harmonic mean score; more info is added if global_precision is used
    '''
    bestModelTargetRecall = None
    bestModelName = None
    bestHM = 0.0

    if global_precision != None:
        #we need to override the accepted and target recall values
        #print(stats.keys())
        lookup_key = list(stats.keys())[0]
        base_total_tp = stats[lookup_key]['RAW_TOTAL_TP']
        base_total_fp = stats[lookup_key]['RAW_TOTAL_FP']
        base_precision = base_total_tp / (base_total_fp + base_total_tp)

        #figure out how far from the goal we are
        delta_precision = global_precision - base_precision
        assert(delta_precision > 0)
        remainder_precision = 1.0 - base_precision

        #now derive our target recall from that difference
        derived_recall = delta_precision / remainder_precision

        #for now, just set both accepted and target to that same value
        acceptedRecall = derived_recall
        targetRecall = derived_recall
    
    #check these after we potentially override any passed in values
    assert(targetRecall >= acceptedRecall)

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
                    if targetRecall == acceptedRecall:
                        #target and accepted are identical, so this is a binary kill switch that's either 1.0 or 0.0
                        if modelRecall >= acceptedRecall:
                            adjRecall = 1.0
                        else:
                            adjRecall = 0.0
                    else:
                        #otherwise, there's a scale
                        adjRecall = (modelRecall - acceptedRecall) / (targetRecall - acceptedRecall)
                        if adjRecall > 1.0:
                            adjRecall = 1.0
                    modelHM = 2 * adjRecall * modelTNR / (adjRecall+modelTNR)
                    
            if modelHM > bestHM:
                bestModelTargetRecall = tr
                bestModelName = mn
                bestHM = modelHM
    
    ret = {
        'eval_recall' : bestModelTargetRecall,
        'model_name' : bestModelName,
        'hm_score' : bestHM
    }
    if global_precision != None:
        ret['base_precision'] = base_precision
        ret['derived_recall'] = derived_recall
    
    return ret

def evaluateVariants(args):
    '''
    This is the work-horse function
    @param args - the command line arguments
    '''
    #open file if we're doing output that way
    if args.outFN != None:
        fp = open(args.outFN, 'wt+')

    #header comes first
    headerValues = [
        ['[Run_Parameters]'],
        ['Models', args.model_directory],
        ['Target_Recall', args.recall],
        ['Model_Mode', args.model],
        ['VCF', args.sample_vcf],
        ['Codicem_Variants', args.codicem],
        ['Raw_Variants', args.variants],
        []
    ]
    for frags in headerValues:
        if args.outFN == None:
            print(*frags, sep='\t')
        else:
            fp.write('\t'.join([str(f) for f in frags])+'\n')
    
    #load the models
    stats, models = loadModels(args.model_directory)

    #run each model (it only prints if the model applies)
    for k in stats.keys():
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
        retLines, retDict = runSubType(reformKey, args, stats[k], models[k], k)
        for frags in retLines:
            if args.outFN == None:
                print(*frags, sep='\t')
            else:
                fp.write('\t'.join([str(f) for f in frags])+'\n')
    
    for k in ['0_3', '1_3']:
        if k not in stats.keys():
            reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
            retLines, retDict = runReferenceCalls(reformKey, args, int(k.split('_')[0]), int(k.split('_')[1]))
            for frags in retLines:
                if args.outFN == None:
                    print(*frags, sep='\t')
                else:
                    fp.write('\t'.join([str(f) for f in frags])+'\n')
    
    if args.outFN != None:
        fp.close()

def runSubType(variantType, args, stats, models, statKey):
    '''
    @param variantType - the variant name
    @param args - the command line arguments
    @param stats - the stats for the variantType
    @param models - the models for the variantType
    @param statKey - the statKey (form '0_1')
    '''
    modelName = args.model
    targetRecall = args.recall

    #overwrite this is the model target doesn't match the passed in target
    modelTargetRecall = args.recall
    
    retLines = []
    retDictForm = []

    #make sure our recall is in the list
    availableRecalls = stats[list(stats.keys())[0]]['ALL_SUMMARY'].keys()
    if targetRecall not in availableRecalls and not ENABLE_AUTO_TARGET:
        raise Exception('Invalid target recall, available options are: %s' % (availableRecalls, ))

    #figure out which models we will actually be using
    if modelName == 'best':
        #best evaluate based on harmonic mean of the overall testing results
        bestModelName = None
        bestHM = 0.0
        for mn in stats.keys():
            #CM = confusion matrix
            modelCM = np.array(stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_CM'][0])
            if (np.sum(modelCM[:, 1]) == 0.0 or np.sum(modelCM[1, :]) == 0):
                modelHM = 0.0
            else:
                #best is harmonic mean of recall and TNR
                modelRecall = modelCM[1, 1] / (modelCM[1, 0] + modelCM[1, 1])
                #modelPrecision = modelCM[1, 1] / (modelCM[0, 1] + modelCM[1, 1])
                modelTNR = modelCM[0, 0] / (modelCM[0, 0] + modelCM[0, 1])
                modelHM = 2 * modelRecall * modelTNR / (modelRecall+modelTNR)
            if modelHM > bestHM or bestModelName == None:
                bestModelName = mn
                bestHM = modelHM
        evalList = [bestModelName]
    
    elif modelName == 'clinical':
        if ENABLE_AUTO_TARGET:
            global_precision = GLOBAL_AUTO_TARGET_PRECISION
            
            #dummy values
            acceptedRecall = 0.0
            targetRecall = 0.0
        else:
            global_precision = None
            #TODO: make this a script parameter instead of hard-coding
            #this contains minimum acceptable recals for a given target
            targetThresholds = {
                '0.995' : 0.99
            }
            if targetRecall not in targetThresholds:
                raise Exception('"clinical" mode has no defined threshold for target recall "%s"' % targetRecall)
            acceptedRecall = targetThresholds[targetRecall]
        
        #get the clinical model
        clinicalModelDict = getClinicalModel(stats, acceptedRecall, targetRecall, global_precision)
        bestModelName = clinicalModelDict['model_name']
        bestModelTargetRecall = clinicalModelDict['eval_recall']
        
        if bestModelName == None:
            #this is the unfortunate event that NO model passes 
            recurseLines, recurseDicts = runReferenceCalls(variantType, args, int(statKey.split('_')[0]), int(statKey.split('_')[1]))
            return retLines+recurseLines, retDictForm+recurseLines
        else:
            modelTargetRecall = bestModelTargetRecall
            evalList = [bestModelName]

    elif modelName == 'all':
        evalList = sorted(stats.keys())
    elif modelName in stats:
        evalList = [modelName]
    else:
        raise Exception('Invalid model name, available options are: %s' % (stats.keys(), ))
    
    #make sure all feature sets are identical and set up the additional field also
    coreFeatureNames = [tuple(f.split('-')) for f in models[evalList[0]]['FEATURES']]
    acceptedVT = models[evalList[0]]['FILTER_VAR_TYPE']
    acceptedGT = models[evalList[0]]['FILTER_CALL_TYPE']
    filtersEnabled = (acceptedVT != -1 or acceptedGT != -1)

    #if the filters are enabled, we need to make sure both were filtered; or stuff will break! :O
    if filtersEnabled:
        assert(acceptedVT != -1 and acceptedGT != -1)
    
    for mn in evalList:
        mnFeatures = [tuple(f.split('-')) for f in models[mn]['FEATURES']]
        assert(coreFeatureNames == mnFeatures)
        assert(acceptedVT == models[mn]['FILTER_VAR_TYPE'])
        assert(acceptedGT == models[mn]['FILTER_CALL_TYPE'])
    
    if filtersEnabled:
        #VT and GT were filtered out, add them back in here so we can filter our variants 
        fields = [('VAR', 'TYPE'), ('CALL', 'GT')] + coreFeatureNames
    else:
        fields = coreFeatureNames
    gtIndex = fields.index(('CALL', 'GT'))

    #load the variant list
    allVariants = []
    if args.variants != None:
        allVariants += parseCLIVariants(args.variants)
    
    if args.codicem != None:
        allVariants += loadCodicemVariants(args.codicem)
    
    #now load the VCF file
    vcf_reader = cyvcf2.VCF(args.sample_vcf)
    raw_reader = cyvcf2.VCF(args.sample_vcf)
    assert(len(vcf_reader.samples) == 1)
    sample_index = 0
    chrom_list = vcf_reader.seqnames

    #go through each variant and extract the features into a shared set
    varIndex = []
    rawVariants = []
    rawGT = []
    varFeatures = []
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        if (chrom not in chrom_list and
            'chr'+chrom in chrom_list):
            chrom = 'chr'+chrom
        
        if chrom in chrom_list:
            variantList = [variant for variant in vcf_reader(f'{chrom}:{start}-{end}')]
        else:
            print('WARNING: Chromosome "%s" not found' % (chrom, ))
            variantList = []
            
        #save the raw variants and which source it is tied to
        rawVariants += variantList
        varIndex += [i]*len(variantList)

        #now go through each variant and pull out the features for it
        for variant in variantList:
            featureVals = getVariantFeatures(variant, sample_index, fields, raw_reader, allowHomRef=True)
            varFeatures.append(featureVals)
            rawGT.append(featureVals[gtIndex])
    
    #if there are no variants, the following will throw errors
    if len(varFeatures) > 0:
        #convert to array 
        varIndex = np.array(varIndex)
        allFeatures = np.array(varFeatures)
        if filtersEnabled:
            #we added ('VAR', 'TYPE'), ('CALL', 'GT') earlier, so remove them now or the model will blow up
            coreFeatures = allFeatures[:, 2:]
        else:
            coreFeatures = allFeatures

        #do the actual predictions now
        prediction = {}
        for mn in evalList:
            clf = models[mn]['MODEL']
            thresh = stats[mn]['ALL_SUMMARY'][modelTargetRecall]['TRAIN_THRESHOLD'][0]
            y_pred_prob = clf.predict_proba(coreFeatures)[:, 1]
            adjPred = ['FP' if y >= thresh else 'TP' for y in y_pred_prob]
            prediction[mn] = adjPred
            #print(mn, adjPred)
    
    #now lets make the actual reporting of things
    header = [
        'chrom', 'start', 'end', 'ref', 'alt', 'call_variant', 'call_gt'
    ]+['%s (%s, %0.4f, %0.4f)' % (mn, modelTargetRecall, stats[mn]['ALL_SUMMARY'][modelTargetRecall]['TEST_TPR'][0], stats[mn]['ALL_SUMMARY'][modelTargetRecall]['TEST_FPR'][0]) for mn in evalList]
    
    retLines.append(['['+variantType+']'])
    retLines.append(header)
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        foundVarIndices = np.where(varIndex == i)[0]
        valList = []
        dictList = []
        if foundVarIndices.shape[0] == 0:
            vals = [chrom, start, end, ref, alt, 'VARIANT_NOT_FOUND', 'VARIANT_NOT_FOUND']+['--']*len(evalList)
            valList.append(vals)

            #if it's not found, we DON'T put it in the dictionary list
        else:
            for ind in foundVarIndices:
                raw_variant_str = generateCyvcf2RecordStr(rawVariants[ind])
                vals = [chrom, start, end, ref, alt, raw_variant_str, rawGT[ind]]
                modelResultDict = {}
                resultsFound = False
                for mn in evalList:
                    if filtersEnabled and (acceptedVT != varFeatures[ind][0] or 
                        acceptedGT != varFeatures[ind][1]):
                        #one of variant type or genotype call don't fit this trained model
                        vals.append('VT_GT_MISMATCH')
                    else:
                        vals.append(prediction[mn][ind])
                        modelString = '%s (%s, %0.4f, %0.4f)' % (mn, modelTargetRecall, stats[mn]['ALL_SUMMARY'][modelTargetRecall]['TEST_TPR'][0], stats[mn]['ALL_SUMMARY'][modelTargetRecall]['TEST_FPR'][0])
                        modelResultDict[modelString] = prediction[mn][ind]
                        resultsFound = True
                
                if(resultsFound):
                    valList.append(vals)

                    #build the dict form
                    d = {
                        'chrom' : chrom,
                        'start' : start,
                        'end' : end,
                        'ref' : ref,
                        'alt' : alt,
                        'call_variant' : raw_variant_str,
                        'call_gt' : rawGT[ind],
                        'predictions' : modelResultDict
                    }
                    dictList.append(d)
        
        for vals in valList:
            #print(*vals, sep='\t')
            retLines.append(vals)

        for d in dictList:
            retDictForm.append(d)

    #print()
    retLines.append([])
    return retLines, retDictForm

def runReferenceCalls(variantType, args, acceptedVT, acceptedGT):
    '''
    TODO
    '''
    retLines = []
    retDictForm = []

    #make sure all feature sets are identical and set up the additional field also
    filtersEnabled = (acceptedVT != -1 or acceptedGT != -1)
    
    if filtersEnabled:
        fields = [('VAR', 'TYPE'), ('CALL', 'GT')]
    gtIndex = fields.index(('CALL', 'GT'))

    #load the variant list
    allVariants = []
    if args.variants != None:
        allVariants += parseCLIVariants(args.variants)
    
    if args.codicem != None:
        allVariants += loadCodicemVariants(args.codicem)
    
    #now load the VCF file
    vcf_reader = cyvcf2.VCF(args.sample_vcf)
    raw_reader = cyvcf2.VCF(args.sample_vcf)
    assert(len(vcf_reader.samples) == 1)
    sample_index = 0
    chrom_list = vcf_reader.seqnames

    #go through each variant and extract the features into a shared set
    varIndex = []
    rawVariants = []
    rawGT = []
    varFeatures = []
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        if (chrom not in chrom_list and
            'chr'+chrom in chrom_list):
            chrom = 'chr'+chrom
        
        if chrom in chrom_list:
            variantList = [variant for variant in vcf_reader(f'{chrom}:{start}-{end}')]
        else:
            print('WARNING: Chromosome "%s" not found' % (chrom, ))
            variantList = []
            
        #save the raw variants and which source it is tied to
        rawVariants += variantList
        varIndex += [i]*len(variantList)

        #now go through each variant and pull out the features for it
        for variant in variantList:
            featureVals = getVariantFeatures(variant, sample_index, fields, raw_reader, allowHomRef=True)
            varFeatures.append(featureVals)
            rawGT.append(featureVals[gtIndex])
    
    #convert to array 
    varIndex = np.array(varIndex)
    
    #now lets make the actual reporting of things
    header = [
        'chrom', 'start', 'end', 'ref', 'alt', 'call_variant', 'call_gt', 'NO_MODEL'
    ]
    
    retLines.append(['['+variantType+']'])
    retLines.append(header)
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        foundVarIndices = np.where(varIndex == i)[0]
        valList = []
        dictList = []
        if foundVarIndices.shape[0] == 0:
            vals = [chrom, start, end, ref, alt, 'VARIANT_NOT_FOUND', 'VARIANT_NOT_FOUND']
            valList.append(vals)
        else:
            for ind in foundVarIndices:
                raw_variant_str = generateCyvcf2RecordStr(rawVariants[ind])
                vals = [chrom, start, end, ref, alt, raw_variant_str, rawGT[ind]]
                modelResultDict = {}
                resultsFound = False
                if filtersEnabled and (acceptedVT != varFeatures[ind][0] or 
                    acceptedGT != varFeatures[ind][1]):
                    #one of variant type or genotype call don't fit this trained model
                    pass
                else:
                    vals.append('FP')
                    modelString = 'NO_TRAINED_MODELS'
                    modelResultDict[modelString] = 'FP'
                    resultsFound = True
                
                if(resultsFound):
                    valList.append(vals)

                    #build the dict form
                    d = {
                        'chrom' : chrom,
                        'start' : start,
                        'end' : end,
                        'ref' : ref,
                        'alt' : alt,
                        'call_variant' : raw_variant_str,
                        'call_gt' : rawGT[ind],
                        'predictions' : modelResultDict
                    }
                    dictList.append(d)
        
        for vals in valList:
            retLines.append(vals)

        for d in dictList:
            retDictForm.append(d)
        
    retLines.append([])
    return retLines, retDictForm
    
def loadModels(modelDir):
    '''
    This will load model information for us
    @param modelDir - the directory with model information
    @return - tuple (stats, models)
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

    return (stats, models)

def parseCLIVariants(csVar):
    '''
    This will parse command line coordinates and return variants
    @param csVar - a comma-separated list of variants of form (chr:start-end)
    @return - a list of variants to evaluate
    '''
    ret = []
    coords = csVar.split(',')
    for coorStr in coords:
        parsed = re.split('[\-:]+', coorStr)
        if len(parsed) == 2:
            chrom = parsed[0]
            start = int(parsed[1])-1 #0-base it
            end = start+1
        elif len(parsed) == 3:
            chrom = parsed[0]
            start = int(parsed[1])-1 #0-base it
            end = int(parsed[2])
        else:
            raise Exception('Unexpected coordinates string: "%s"' % coorStr)
        var = (chrom, start, end, '--', '--')
        if (var not in ret):
            ret.append(var)
    return ret

def generateCyvcf2RecordStr(variant):
    '''
    Simple cyvcf2 variant reformatter, makes it match something similar to vcf
    @param variant - a variant from cyvcf2
    @return - a string formatted as a "record"
    '''
    chrom = variant.CHROM
    pos = variant.POS
    ref = variant.REF
    alts = variant.ALT
    return f'cyvcf2.record(CHROM={chrom}, POS={pos}, REF={ref}, ALT={alts})'

def loadCodicemVariants(csvFN):
    '''
    Loads a Codicem sanger CSV and finds variants that need confirmation
    @param csvFN - the filename to read from
    @return - a list of variants to evaluate
    '''
    ret = []
    fp = open(csvFN, 'r')
    csvReader = csv.DictReader(fp)
    for d in csvReader:
        chrom = d['Chromosome']
        start = int(d['Start'])-1 #need to 0-base this
        end = int(d['Stop'])
        ref = d['Reference']
        alt = d['Alternate']
        var = (chrom, start, end, ref, alt)
        if (var not in ret):
            ret.append(var)
    fp.close()
    return ret

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for evaluating variants for false positive status"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-c', '--codicem', dest='codicem', default=None, help='a Codicem CSV file with variants to evaluate (default: None)')
    p.add_argument('-v', '--variants', dest='variants', default=None, help='variant coordinates to evaluate (default: None)')
    p.add_argument('-m', '--model', dest='model', default='clinical', help='the model name to use (default: clinical)')
    p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')
    p.add_argument('-o', '--output', dest='outFN', default=None, help='the place to send output to (default: stdout)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    p.add_argument('sample_vcf', type=str, help='VCF file with variants to evaulate')

    #parse the arguments
    args = p.parse_args()

    evaluateVariants(args)
