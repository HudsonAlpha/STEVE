
import argparse as ap
import csv
import os
import vcf

def parseSlids(slidStr):
    '''
    This will parse a list of SL##s 
    @param slidStr - a string containing comma separated slids; "-" is also accepted for ranges; 
        ex: "SL123456-SL123467,SL333333"; it can also be a filename with same info on lines
    @return - a list of all slids we need to pull data for
    '''
    ret = []
    if os.path.exists(slidStr):
        #load a file with lines
        fp = open(slidStr, 'rt')
        frags = []
        for l in fp:
            frags.append(l.rstrip())
        fp.close()
    else:
        #load as fragments
        frags = slidStr.split(',')

    for fragment in frags:
        subFrags = fragment.split('-')
        assert(len(subFrags) <= 2)
        if len(subFrags) == 1:
            #single slid
            assert(subFrags[0][0:2] == 'SL')
            ret.append(subFrags[0])
        elif len(subFrags) == 2:
            #range of slids
            r1 = subFrags[0]
            r2 = subFrags[1]
            assert(r1[0:2] == 'SL' and r2[0:2] == 'SL')
            i1 = int(r1[2:])
            i2 = int(r2[2:])
            assert(i2 >= i1)
            for i in range(i1, i2+1):
                ret.append('SL'+str(i))
        else:
            raise Exception('How did you get here?')
    return ret

def loadVariantFile(fn, outFN):
    '''
    '''
    vcfReader = vcf.Reader(filename=fn, compressed=fn.endswith('.gz'))
    fpo = open(outFN, 'w+')
    fpo.write(','.join(['Chromosome', 'Start', 'Stop', 'Reference', 'Alternate'])+'\n')
    variants = []
    for variant in vcfReader:
        var = ('chr'+variant.CHROM, str(variant.POS), str(variant.POS), variant.REF, ';'.join([str(s) for s in variant.ALT]))
        fpo.write(','.join(var)+'\n')
        variants.append(var)
    fpo.close()
    return variants

def runCostAnalysis(slid, variantFN):
    resultFN = '/gpfs/gpfs1/home/jholt/sanger_less_tests/results/cost_results_v2/%s.tsv' % slid
    if not os.path.exists(resultFN):
        cmdFrags = [
            'python3', '-u', 'EvaluateVariants.py',
            '-c', variantFN,
            '-m', 'clinical_v2',
            '-r', '0.995',
            '-o', resultFN,
            '/gpfs/gpfs1/home/jholt/sanger_less_tests/clinical_models/dragen-clinical-v1.0.0',
            '/gpfs/gpfs2/dragen/clinical/v0/%s/%s.hard-filtered.gvcf.gz' % (slid, slid)
        ]
        cmd = ' '.join(cmdFrags)
        print(cmd)
        os.system(cmd)
    else:
        print('%s exists.' % resultFN)
        pass
    return resultFN

def loadResults(resultFN):
    fp = open(resultFN, 'r')
    lines = []
    blocks = {}
    for l in fp:
        if l[0] == '[':
            if len(lines) > 0:
                blocks[category] = lines
            category = l.rstrip()[1:-1]
            lines = []
        else:
            lines.append(l.rstrip())
    fp.close()

    ret = {}
    for label in blocks:
        if label == 'Run_Parameters':
            continue
        var_type, call_type = label.split('_')
        csvReader = csv.DictReader(blocks[label], delimiter='\t')
        for d in csvReader:
            chrom = d['chrom']
            pos = d['end']
            ref = d['ref']
            alt = d['alt']

            if d['call_variant'] == 'VARIANT_NOT_FOUND':
                ret[(chrom, pos, ref, alt)] = ('VARIANT_NOT_FOUND', 'VARIANT_NOT_FOUND', '--')
            else:
                pred = None
                for k in d:
                    if d[k] == 'TP' or d[k] == 'FP':
                        assert(pred == None)
                        pred = d[k]
                assert(pred != None)
                ret[(chrom, pos, ref, alt)] = (var_type, call_type, pred)
    
    return ret
    
if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for costing the expected variant price per case"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    p.add_argument('-v', '--variants', dest='variants', default=None, help='variant coordinates to evaluate (default: None)')
    #p.add_argument('-m', '--model', dest='model', default='best', help='the model name to use (default: best)')
    #p.add_argument('-r', '--recall', dest='recall', default='0.995', help='the target recall value from training (default: 0.995)')
    #p.add_argument('-o', '--output', dest='outFN', default=None, help='the place to send output to (default: stdout)')

    #required main arguments
    p.add_argument('model_directory', type=str, help='directory with models and model stats')
    #p.add_argument('sample_vcf', type=str, help='VCF file with variants to evaulate')
    PATTERN = '/gpfs/gpfs2/dragen/clinical/v0/{slid}/{slid}.hard-filtered.gvcf.gz'
    p.add_argument('slids', type=str, help='the list of slids separate by commas (ex: "SL123456-SL123467,SL333333")')

    #parse the arguments
    args = p.parse_args()

    #load the variant list
    variants = loadVariantFile(args.variants, args.variants+'.csv')

    #load the samples
    sampleList = sorted(set(parseSlids(args.slids)))
    #sampleList = sampleList[0:10]
    #print(sampleList)

    countedCategories = [
        'SNP_HET', 'SNP_HOM', 'SNP_HE2', 
        'INDEL_HET', 'INDEL_HOM', 'INDEL_HE2'
    ]

    variantResults = {}
    for slid in sampleList:
        resultFN = runCostAnalysis(slid, args.variants+'.csv')
        resultDict = loadResults(resultFN)
        
        #for varKey in resultDict:
        for variant in variants:
            varKey = (variant[0], variant[1], variant[3], variant[4])
            if varKey not in variantResults:
                variantResults[varKey] = {}
            
            if varKey in resultDict:
                varType, callType, pred = resultDict[varKey]
                nk = varType+'_'+callType

                if nk in countedCategories:
                    #if nk not in variantResults[varKey]:
                    #    variantResults[varKey][nk] = {}
                    storeKey = callType+'_'+pred
                    variantResults[varKey][storeKey] = variantResults[varKey].get(storeKey, 0)+1

    keyLookups = [
        'HET_TP', 'HET_FP',
        'HOM_TP', 'HOM_FP',
        'HE2_TP', 'HE2_FP'
    ]
    ignoreVariants = set([
        ('chr13', '16000147', 'T', 'C'),
        ('chr15', '17000120', 'G', 'T'),
        ('chr22', '10511391', 'T', 'A')
    ])
    print('VARIANT(N=%d)' % len(sampleList), *keyLookups, sep='\t')
    for k in sorted(variantResults):
        if (k[0] in ['chrM', 'chrY']) or (int(k[1]) < 100000) or (k in ignoreVariants):
            continue
        arrVals = []
        for kl in keyLookups:
            arrVals.append(variantResults.get(k, {}).get(kl, 0))

        print(k, *arrVals, sep='\t')