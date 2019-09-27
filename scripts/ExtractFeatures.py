
import argparse as ap
import datetime
import json
import numpy as np
import os
import re
import vcf

from RunTrainingPipeline import parseSlids

#Missing value
DEFAULT_MISSING = -1000.0

#Variant zygosity constants
GT_REF_HET = 0
GT_ALT_HOM = 1
GT_HET_HET = 2
GT_REF_HOM = 3
GT_TRANSLATE = {
    GT_REF_HET : 'HET',
    GT_ALT_HOM : 'HOM',
    GT_HET_HET : 'HE2',
    GT_REF_HOM : 'HOMREF'
}

#Variant type constants
VAR_SNP = 0
VAR_INDEL = 1

#this file contains all the metrics we care about for each variant caller
METRICS_FN = '/gpfs/gpfs1/home/jholt/sanger_less_tests/scripts/model_metrics.json'
fp = open(METRICS_FN, 'r')
UNPARSED_METRICS = json.load(fp)
fp.close()

#we need to parse it by defined & copied sets
ALL_METRICS = {}
for k in UNPARSED_METRICS['defined']:
    ALL_METRICS[k] = UNPARSED_METRICS['defined'][k]
for k in UNPARSED_METRICS['copied']:
    assert(k not in ALL_METRICS)
    ALL_METRICS[k] = ALL_METRICS[UNPARSED_METRICS['copied'][k]]

def getVariantFeatures(variant, sampleID, fields, rawReader):
    '''
    This function takes a variant from a VCF and a sample ID and extracts the features for it
    @param variant - the variant (from "vcfReader")
    @param sampleID - the sample in the VCF file
    @param fields - the fields to extract from the variant
    @param rawReader - the raw VCF file
    @return - a list of ordered features
    '''
    annots = []

    #first, annotate whether it's a SNP or an indel
    callStats = variant.genotype(sampleID)
    gtPieces = re.split('[/|]', callStats['GT'])
    if len(gtPieces) != 2:
        #TODO: only a problem in strelka, how to handle it?
        return None
    assert(gtPieces[0] != '0' or gtPieces[1] != '0')
    
    #look at the variant call
    s1 = (variant.REF if gtPieces[0] == '0' else variant.ALT[int(gtPieces[0])-1])
    s2 = (variant.REF if gtPieces[1] == '0' else variant.ALT[int(gtPieces[1])-1])
    if str(s1) == '<NON_REF>' or str(s2) == '<NON_REF>':
        #ignore this one, it's something we can't figure out from a GVCF
        #TODO: do we feel like ever changing this?
        return None
    
    #pull this out for use by all the AD stat fields
    callAD = callStats['AD']
    adSum = sum(callAD)
    
    #run the top two filters
    #maxAF = (max([adVal / adSum for adVal in callAD[1:]]) if adSum != 0.0 else 0.0)
    #if adSum < 8 or maxAF < 0.15:
    #    continue

    for k, subk in fields:
        if k == 'VAR':
            if subk == 'TYPE':
                #NOTE: the last clause is to fix the following issue:
                # if the variant is 1/2 and REF=AC but ALT=A,C then it's a double del and should be marked as such
                #print(variant.CHROM, variant.POS, variant.REF, variant.ALT, gtPieces, s1, s2)
                if len(s1) == 1 and len(s2) == 1 and len(variant.REF) == 1:
                    annots.append(VAR_SNP)
                else:
                    annots.append(VAR_INDEL)
            else:
                raise Exception('Unexpected metric key: %s-%s' % (k, subk))

        elif k == 'CALL':
            #these are call specific measures
            if subk == 'GT':
                if gtPieces[0] == gtPieces[1]:
                    if gtPieces[0] == '0':
                        val = GT_REF_HOM
                    else:
                        val = GT_ALT_HOM
                elif gtPieces[0] == '0' or gtPieces[1] == '0':
                    val = GT_REF_HET
                else:
                    val = GT_HET_HET
            elif subk == 'AD0':
                val = (callAD[int(gtPieces[0])] if adSum > 0 else DEFAULT_MISSING)
            elif subk == 'AD1':
                val = (callAD[int(gtPieces[1])] if adSum > 0 else DEFAULT_MISSING)
            elif subk == 'ADO':
                if gtPieces[0] == gtPieces[1]:
                    #homozygous, pull AD once
                    adUsed = callAD[int(gtPieces[0])]
                else:
                    #heterozygous, get both AD vals
                    adUsed = callAD[int(gtPieces[0])] + callAD[int(gtPieces[1])]
                
                #AD-other is the total AD minus the GT AD
                val = ((adSum - adUsed) if adSum > 0 else DEFAULT_MISSING)
            elif subk == 'AF0':
                val = (callAD[int(gtPieces[0])] / adSum if adSum > 0 else DEFAULT_MISSING)
            elif subk == 'AF1':
                val = (callAD[int(gtPieces[1])] / adSum if adSum > 0 else DEFAULT_MISSING)
            elif subk == 'AFO':
                if gtPieces[0] == gtPieces[1]:
                    #homozygous, pull AD once
                    adUsed = callAD[int(gtPieces[0])]
                else:
                    #heterozygous, get both AD vals
                    adUsed = callAD[int(gtPieces[0])] + callAD[int(gtPieces[1])]
                
                #AD-other is the total AD minus the GT AD
                val = ((adSum - adUsed) / adSum if adSum > 0 else DEFAULT_MISSING)
            else:
                try:
                    val = float(callStats[subk])
                except AttributeError as e:
                    #TODO: is this okay for everything?
                    val = DEFAULT_MISSING
            annots.append(val)

        elif k == 'INFO':
            #get from the INFO column
            val = (float(variant.INFO[subk]) if (subk in variant.INFO) else DEFAULT_MISSING)
            annots.append(val)
        
        elif k == 'MUNGED':
            if subk == 'DP_DP':
                try:
                    dpCall = callStats['DP']
                    dpVar = variant.INFO['DP']
                    dpRat = dpCall / dpVar
                    annots.append(dpRat)
                except:
                    #TODO: this was added for strelka, should we remove or patch it better?
                    annots.append(DEFAULT_MISSING)
            elif subk == 'QUAL':
                val = variant.QUAL
                annots.append(val)
            elif subk == 'NEARBY':
                #search the raw VCF for nearby calls
                nearbyFound = 0
                BUFFER = 20
                for rawVariant in rawReader.fetch(variant.CHROM, variant.POS-BUFFER, variant.POS+BUFFER):
                    if rawVariant.POS != variant.POS:
                        rawData = rawVariant.genotype(sampleID)
                        if rawData['GT'] not in ['./.', '0/0']:
                            nearbyFound += 1
                            
                annots.append(nearbyFound)
            elif subk == 'FILTER':
                #store the number of filters applied (PASS doesn't show up in these lists)
                annots.append(len(variant.FILTER))
            else:
                raise Exception('Unhandled computed measurement: %s-%s' % (k, subk))
        else:
            raise Exception('Unexpected metric key: %s' % (k, ))

    return annots

def gatherVcfMetrics(vcfFN, rawVCF, metrics):
    '''
    This will gather a specific set of metrics from an RTG vcfeval output file
    @param vcfFN - the vcf filename to load stats for, limited to one rtg output type (i.e. tp, fp, etc.)
    @param rawVCF - the original vcf filename
    @param metrics - the specific set of metrics we want to get info on; key is the broad category of where the data
    is in the VCF, and value is a list of values from that category to gather
    @return - tuple (data, fields)
        data - a NxF matrix of variant annotations; N = number of variants, F = number of fields per variant
        fields - the column labels of the F dimension (i.e. field names)
    '''
    #list of variants
    ret = []
    fields = []

    #open up the VCF file
    vcfReader = vcf.Reader(filename=vcfFN, compressed=True)
    rawReader = vcf.Reader(filename=rawVCF, compressed=True)

    #get the sample
    samples = vcfReader.samples
    assert(len(samples) == 1)
    sampleID = samples[0]
    
    #fill out the fields; always include a variant type field
    fields.append(('VAR', 'TYPE'))

    #sort the keys and subkeys of the remaining fields to make sure everything happens in the same order
    metricOrder = sorted(metrics.keys())
    for mo in metricOrder:
        metrics[mo].sort()
        for subk in metrics[mo]:
            fields.append((mo, subk))
    
    #now we go through
    for variant in vcfReader:
        annots = getVariantFeatures(variant, sampleID, fields, rawReader)
        if annots != None:
            ret.append(annots)
            if len(ret) % 100000 == 0:
                print('[%s] Processed %d variants -> (%s:%s %s %s)' % (datetime.datetime.now(), len(ret), variant.CHROM, variant.POS, variant.REF, variant.ALT))
                #break

    return ret, fields

def extractFeatures(aligner, caller, samples, outPrefix):
    #0 - constants based on input
    RTG_ROOT = '/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/rtg_results/%s/%s' % (aligner, caller)
    VCF_ROOT = '/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/variant_calls/%s/%s' % (aligner, caller)
    #OUTPUT_ROOT = '/gpfs/gpfs1/home/jholt/sanger_less_tests/results/rtgReformed'
    OUTPUT_ROOT = outPrefix
    CLASSIFICATION_TYPES = ['fp', 'tp']
    VCF_METRICS = ALL_METRICS[caller]
    
    #TODO: make regenerate a parameter
    REGENERATE = True
    
    #1 - load the samples
    SAMPLES = parseSlids(samples)
    #fp = open(sampleFN, 'r')
    #for l in fp:
    #    SAMPLES.append(l.rstrip())
    #fp.close()
    
    #2 - make sure results exist for each aligner-caller-sample triplet
    acDir = '%s/%s/%s' % (OUTPUT_ROOT, aligner, caller)
    if not os.path.exists(acDir):
        os.makedirs(acDir)
    
    missingFiles = []
    for sample in SAMPLES:
        for ct in CLASSIFICATION_TYPES:
            rtgVcfFN = '%s/%s/%s.vcf.gz' % (RTG_ROOT, sample, ct)
            rawVCF = '%s/%s.vcf.gz' % (VCF_ROOT, sample)
            if not os.path.exists(rtgVcfFN):
                missingFiles.append(rtgVcfFN)
            if not rawVCF:
                missingFiles.append(rawVCF)
    
    if len(missingFiles) > 0:
        print('Missing files:', missingFiles)
        raise Exception('Files required to perform the analysis are missing')

    #3 - reformat results into feature sets as necessary
    for sample in SAMPLES:
        for ct in CLASSIFICATION_TYPES:
            outNumpyFN = '%s/%s/%s/%s_%s.npy' % (OUTPUT_ROOT, aligner, caller, sample, ct)
            outCatFN = '%s/%s/%s/%s_%s_fields.json' % (OUTPUT_ROOT, aligner, caller, sample, ct)
            if not os.path.exists(outNumpyFN) or REGENERATE:
                print('Building "%s"...' % (outNumpyFN, ))
                rtgVcfFN = '%s/%s/%s.vcf.gz' % (RTG_ROOT, sample, ct)
                rawVCF = '%s/%s.vcf.gz' % (VCF_ROOT, sample)
                results, fields = gatherVcfMetrics(rtgVcfFN, rawVCF, VCF_METRICS)
                numpyDat = np.array(results)
                
                #save the data points
                print('Saving %s data points to %s' % (str(numpyDat.shape), outNumpyFN))
                np.save(outNumpyFN, numpyDat)

                #also save the field file with it
                fp = open(outCatFN, 'w+')
                json.dump(['-'.join(tup) for tup in fields], fp)
                fp.close()

    #we will build a list of numpy arrays, then stack at the end
    tpList = []
    fpList = []

    #load the fields from the first file
    fp = open('%s/%s/%s/%s_tp_fields.json' % (OUTPUT_ROOT, aligner, caller, SAMPLES[0]))
    fieldsList = json.load(fp)
    fp.close()
    
    #get data from each sample
    for sample in SAMPLES:
        #TP first
        tpFN = '%s/%s/%s/%s_tp.npy' % (OUTPUT_ROOT, aligner, caller, sample)
        tpOrder = '%s/%s/%s/%s_tp_fields.json' % (OUTPUT_ROOT, aligner, caller, sample)
        fp = open(tpOrder, 'r')
        tpFields = json.load(fp)
        fp.close()
        assert(tpFields == fieldsList) #if these fails, the data was gathered with different models, need to re-run
        tpVar = np.load(tpFN, 'r')
        tpList.append(tpVar)

        #now false positives
        fpFN = '%s/%s/%s/%s_fp.npy' % (OUTPUT_ROOT, aligner, caller, sample)
        fpOrder = '%s/%s/%s/%s_fp_fields.json' % (OUTPUT_ROOT, aligner, caller, sample)
        fp = open(fpOrder, 'r')
        fpFields = json.load(fp)
        fp.close()
        assert(fpFields == fieldsList) #if these fails, the data was gathered with different models, need to re-run
        fpVar = np.load(fpFN, 'r')
        fpList.append(fpVar)
    
    tpLens = [arr.shape[0] for arr in tpList]
    fpLens = [arr.shape[0] for arr in fpList]

    print(tpLens)
    print(fpLens)
    #return
    
    #These steps will go into a training.py file
    #allTP = np.vstack(tpList)
    #allFP = np.vstack(fpList)
    #testClassifier(allTP, allFP, fieldsList, OUTPUT_ROOT)

if __name__ == "__main__":
    #first set up the arg parser
    DESC="Scripts for extracting variant features from a VCF file"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-d', '--date-subdir', dest='date_subdir', default=None, help='the date subdirectory (default: "hli-YYMMDD")')
    
    #required main arguments
    p.add_argument('aligner', type=str, help='the aligner to train on')
    p.add_argument('caller', type=str, help='the variant caller to train on')
    p.add_argument('sample_metadata', type=str, help='a file with sample identifiers')
    p.add_argument('output_prefix', type=str, help='prefix to save output values to')

    #parse the arguments
    args = p.parse_args()

    extractFeatures(args.aligner, args.caller, args.sample_metadata, args.output_prefix)