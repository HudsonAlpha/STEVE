
import cyvcf2
import glob
import numpy as np

def getDeltas(fn):
    vcf = cyvcf2.VCF(fn)
    dists = []
    prevChrom = ''
    prevPos = 0
    for variant in vcf:
        if prevChrom == variant.CHROM:
            dists.append(variant.POS - prevPos)
            prevPos = variant.POS
        else:
            prevChrom = variant.CHROM
            prevPos = variant.POS
    return np.array(dists)

if __name__ == "__main__":
    #aligner = 'dragen-07.011.352.3.2.8b'
    #caller = 'dragen-07.011.352.3.2.8b'
    
    aligner = 'dragen-07.021.510.3.5.7'
    caller = 'dragen-07.021.510.3.5.7'

    #aligner = 'clinical_sentieon-201808.07'
    #caller = 'strelka-2.9.10'
    
    print(f'/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/rtg_results/{aligner}/{caller}/SL*/fp.vcf.gz')
    vcfList = sorted(glob.glob(f'/gpfs/gpfs1/home/jholt/csl_validations/core_pipeline_analysis/pipeline/rtg_results/{aligner}/{caller}/SL*/fp.vcf.gz'))
    print(vcfList)
    for fn in vcfList:
        print(fn)
        results = getDeltas(fn)
        #print(results)
        for x in range(1, 5):
            counts = np.sum(results < 10**x)
            print('', 10**x, counts, 100*counts / results.shape[0], sep='\t')
