
import argparse as ap
import glob
import json
import numpy as np

from ExtractFeatures import GT_TRANSLATE, VAR_TRANSLATE

def summarizeFeatures(featureDir):
    '''
    @param featureDir - a directory with _fp.npy and _tp.npy files (and field jsons)
    @return - None, but will print summary stats for all data
    '''
    fpFiles = sorted(glob.glob('%s/*_fp.npy' % featureDir))
    tpFiles = sorted(glob.glob('%s/*_tp.npy' % featureDir))
    assert(len(fpFiles) == len(tpFiles))

    fpDict = {
        'total' : {}
    }
    for fn in fpFiles:
        slid = fn.split('/')[-1].strip('_fp.npy')
        stats = getStats(fn)
        #print(slid, 'fp', stats)
        fpDict[slid] = stats
        for k in stats:
            fpDict['total'][k] = fpDict['total'].get(k, 0)+stats[k]
    
    printStats('FP', fpDict)
    print()
    
    tpDict = {
        'total' : {}
    }
    for fn in tpFiles:
        slid = fn.split('/')[-1].strip('_tp.npy')
        stats = getStats(fn)
        #print(slid, 'tp', stats)
        tpDict[slid] = stats

        for k in stats:
            tpDict['total'][k] = tpDict['total'].get(k, 0)+stats[k]
    
    printStats('TP', tpDict)
    print()

    assert(fpDict.keys() == tpDict.keys())

def getStats(fn):
    fieldFN = fn[:-4]+'_fields.json'
    fp = open(fieldFN, 'r')
    fields = json.load(fp)
    fp.close()
    
    varInd = fields.index('VAR-TYPE')
    gtInd = fields.index('CALL-GT')
    
    dat = np.load(fn, 'r')
    ret = {}
    for row in dat:
        varVal = row[varInd]
        gtVal = row[gtInd]
        ret[(varVal, gtVal)] = ret.get((varVal, gtVal), 0)+1
    return ret

def printStats(tfLabel, allStats):
    header = ['sample', 'TP/FP']
    for vt in range(0, 2):
        for gt in range(0, 3):
            header.append(VAR_TRANSLATE[vt]+'-'+GT_TRANSLATE[gt])
    print(*header, sep='\t')

    for k in sorted(allStats.keys()):
        row = [k, tfLabel]
        for vt in range(0, 2):
            for gt in range(0, 3):
                row.append(str(allStats[k].get((vt, gt), 0)))
        print(*row, sep='\t')


if __name__ == "__main__":
    #first set up the arg parser
    DESC="Script for summarizing feature statistics"
    p = ap.ArgumentParser(description=DESC, formatter_class=ap.RawTextHelpFormatter)
    
    #optional arguments with default
    #p.add_argument('-r', '--recall', dest='recall', default='0.99', help='the target recall value from training (default: 0.99)')
    
    #required main arguments
    p.add_argument('features_directory', type=str, help='directory with features')
    
    #parse the arguments
    args = p.parse_args()

    summarizeFeatures(args.features_directory)