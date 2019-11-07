
import argparse as ap
import csv
import json
import numpy as np
import pickle
import vcf

from ExtractFeatures import ALL_METRICS, getVariantFeatures, GT_TRANSLATE, VAR_TRANSLATE

def evaluateVariants(args):
    '''
    This is the work-horse function
    @param args - the command line arguments
    '''
    #load the models
    stats, models = loadModels(args.model_directory)
    #print(json.dumps(stats, indent=4))

    #run each model (it only prints if the model applies)
    for k in stats.keys():
        reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
        runSubType(reformKey, args, stats[k], models[k])
    
    for k in ['0_3', '1_3']:
        if k not in stats.keys():
            reformKey = VAR_TRANSLATE[int(k.split('_')[0])]+'_'+GT_TRANSLATE[int(k.split('_')[1])]
            runReferenceCalls(reformKey, args, int(k.split('_')[0]), int(k.split('_')[1]))

def runSubType(variantType, args, stats, models):
    '''
    @param variantType - the variant name
    @param args - the command line arguments
    @param stats - the stats for the variantType
    @param models - the models for the variantType
    '''
    modelName = args.model
    targetRecall = args.recall

    #make sure our recall is in the list
    availableRecalls = stats[list(stats.keys())[0]]['ALL_SUMMARY'].keys()
    if targetRecall not in availableRecalls:
        raise Exception('Invalid target recall, available options are: %s' % (availableRecalls, ))

    #figure out which models we will actually be using
    if modelName == 'best':
        bestModelName = None
        #bestFPR = 1.0
        bestHM = 0.0
        for mn in stats.keys():
            #modelFPR = stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_FPR'][0]
            #if modelFPR < bestFPR:
                #bestModelName = mn
                #bestFPR = modelFPR
            modelCM = np.array(stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_CM'][0])
            if (np.sum(modelCM[:, 1]) == 0.0 or np.sum(modelCM[1, :]) == 0):
                modelHM = 0.0
            else:
                modelRecall = modelCM[1, 1] / (modelCM[1, 0] + modelCM[1, 1])
                modelPrecision = modelCM[1, 1] / (modelCM[0, 1] + modelCM[1, 1])
                modelHM = 2 * modelRecall * modelPrecision * (modelRecall+modelPrecision)
            if modelHM > bestHM or bestModelName == None:
                bestModelName = mn
                bestHM = modelHM
            
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
    
    for mn in evalList:
        mnFeatures = [tuple(f.split('-')) for f in models[mn]['FEATURES']]
        assert(coreFeatureNames == mnFeatures)
        assert(acceptedVT == models[mn]['FILTER_VAR_TYPE'])
        assert(acceptedGT == models[mn]['FILTER_CALL_TYPE'])
    
    if filtersEnabled:
        fields = [('VAR', 'TYPE'), ('CALL', 'GT')] + coreFeatureNames
    else:
        fields = coreFeatureNames
    gtIndex = fields.index(('CALL', 'GT'))

    #load the variant list
    allVariants = []
    if args.variants != None:
        raise Exception('No impl')
    
    if args.codicem != None:
        allVariants += loadCodicemVariants(args.codicem)
    
    #now load the VCF file
    vcfReader = vcf.Reader(filename=args.sample_vcf, compressed=True)
    rawReader = vcf.Reader(filename=args.sample_vcf, compressed=True)
    assert(len(vcfReader.samples) == 1)
    chromList = vcfReader.contigs.keys()

    #go through each variant and extract the features into a shared set
    varIndex = []
    rawVariants = []
    rawGT = []
    varFeatures = []
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        if (chrom not in chromList and
            'chr'+chrom in chromList):
            chrom = 'chr'+chrom
        
        if chrom in chromList:
            variantList = [variant for variant in vcfReader.fetch(chrom, start, end)]
        else:
            print('Chromosome "%s" not found' % (chrom, ))
            variantList = []
            
        #save the raw variants and which source it is tied to
        rawVariants += variantList
        varIndex += [i]*len(variantList)

        #now go through each variant and pull out the features for it
        for variant in variantList:
            featureVals = getVariantFeatures(variant, vcfReader.samples[0], fields, rawReader, allowHomRef=True)
            varFeatures.append(featureVals)
            rawGT.append(featureVals[gtIndex])
    
    #convert to array 
    varIndex = np.array(varIndex)
    allFeatures = np.array(varFeatures)
    if filtersEnabled:
        coreFeatures = allFeatures[:, 2:]
    else:
        coreFeatures = allFeatures

    #do the actual predictions now
    prediction = {}
    for mn in evalList:
        clf = models[mn]['MODEL']
        thresh = stats[mn]['ALL_SUMMARY'][targetRecall]['TRAIN_THRESHOLD'][0]
        y_pred_prob = clf.predict_proba(coreFeatures)[:, 1]
        adjPred = ['FP' if y >= thresh else 'TP' for y in y_pred_prob]
        prediction[mn] = adjPred
        #print(mn, adjPred)
    
    #now lets make the actual reporting of things
    header = [
        'chrom', 'start', 'end', 'ref', 'alt', 'call_variant', 'call_gt'
    ]+['%s (%0.4f, %0.4f)' % (mn, stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_TPR'][0], stats[mn]['ALL_SUMMARY'][targetRecall]['TEST_FPR'][0]) for mn in evalList]
    
    print(variantType)
    print(*header, sep='\t')
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        foundVarIndices = np.where(varIndex == i)[0]
        valList = []
        if foundVarIndices.shape[0] == 0:
            vals = [chrom, start, end, ref, alt, 'VARIANT_NOT_FOUND', 'VARIANT_NOT_FOUND']+['--']*len(evalList)
            valList.append(vals)
        else:
            for ind in foundVarIndices:
                vals = [chrom, start, end, ref, alt, rawVariants[ind], rawGT[ind]]
                resultsFound = False
                for mn in evalList:
                    if filtersEnabled and (acceptedVT != varFeatures[ind][0] or 
                        acceptedGT != varFeatures[ind][1]):
                        #one of variant type or genotype call don't fit this trained model
                        vals.append('VT_GT_MISMATCH')
                    else:
                        vals.append(prediction[mn][ind])
                        resultsFound = True
                
                if(resultsFound):
                    valList.append(vals)
        
        for vals in valList:
            print(*vals, sep='\t')
    print()

def runReferenceCalls(variantType, args, acceptedVT, acceptedGT):
    '''
    TODO
    '''
    #make sure all feature sets are identical and set up the additional field also
    filtersEnabled = (acceptedVT != -1 or acceptedGT != -1)
    
    if filtersEnabled:
        fields = [('VAR', 'TYPE'), ('CALL', 'GT')]
    gtIndex = fields.index(('CALL', 'GT'))

    #load the variant list
    allVariants = []
    if args.variants != None:
        raise Exception('No impl')
    
    if args.codicem != None:
        allVariants += loadCodicemVariants(args.codicem)
    
    #now load the VCF file
    vcfReader = vcf.Reader(filename=args.sample_vcf, compressed=True)
    rawReader = vcf.Reader(filename=args.sample_vcf, compressed=True)
    assert(len(vcfReader.samples) == 1)
    chromList = vcfReader.contigs.keys()

    #go through each variant and extract the features into a shared set
    varIndex = []
    rawVariants = []
    rawGT = []
    varFeatures = []
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        if (chrom not in chromList and
            'chr'+chrom in chromList):
            chrom = 'chr'+chrom
        
        if chrom in chromList:
            variantList = [variant for variant in vcfReader.fetch(chrom, start, end)]
        else:
            print('Chromosome "%s" not found' % (chrom, ))
            variantList = []
            
        #save the raw variants and which source it is tied to
        rawVariants += variantList
        varIndex += [i]*len(variantList)

        #now go through each variant and pull out the features for it
        for variant in variantList:
            featureVals = getVariantFeatures(variant, vcfReader.samples[0], fields, rawReader, allowHomRef=True)
            varFeatures.append(featureVals)
            rawGT.append(featureVals[gtIndex])
    
    #convert to array 
    varIndex = np.array(varIndex)
    
    #now lets make the actual reporting of things
    header = [
        'chrom', 'start', 'end', 'ref', 'alt', 'call_variant', 'call_gt'
    ]
    
    print(variantType)
    print(*header, sep='\t')
    for i, (chrom, start, end, ref, alt) in enumerate(allVariants):
        foundVarIndices = np.where(varIndex == i)[0]
        valList = []
        if foundVarIndices.shape[0] == 0:
            vals = [chrom, start, end, ref, alt, 'VARIANT_NOT_FOUND', 'VARIANT_NOT_FOUND']
            valList.append(vals)
        else:
            for ind in foundVarIndices:
                vals = [chrom, start, end, ref, alt, rawVariants[ind], rawGT[ind]]
                resultsFound = False
                if filtersEnabled and (acceptedVT != varFeatures[ind][0] or 
                    acceptedGT != varFeatures[ind][1]):
                    #one of variant type or genotype call don't fit this trained model
                    pass
                else:
                    vals.append('FP')
                    resultsFound = True
                
                if(resultsFound):
                    valList.append(vals)
        
        for vals in valList:
            print(*vals, sep='\t')
    print()
    
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
    p.add_argument('-m', '--model', dest='model', default='best', help='the model name to use (default: best)')
    p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    p.add_argument('sample_vcf', type=str, help='VCF file with variants to evaulate')

    #parse the arguments
    args = p.parse_args()

    evaluateVariants(args)