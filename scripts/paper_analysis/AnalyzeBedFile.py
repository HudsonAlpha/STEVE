
#import argparse as ap
import cyvcf2
import glob
import os

def findEmptyRegions(vcfFN, bedFN, filteredFN):
    print(vcfFN)
    print(bedFN)
    vcf = cyvcf2.VCF(vcfFN)

    fp = open(bedFN, 'r')
    fpo = open(filteredFN, 'w+')
    found = 0
    empty = 0
    for l in fp:
        fields = l.rstrip().split('\t')
        chrom, start, end = fields

        c = False
        for var in vcf(f'{chrom}:{start}-{end}'):
            c = True
            break
        if c:
            #print(*fields, sep='\t')
            found += 1
            fpo.write(l)
        else:
            empty += 1
    
    fpo.close()
    fp.close()
    print(found, empty, sep='\t')

if __name__ == "__main__":
    vcf1 = '/gpfs/gpfs1/home/jholt/sanger_less_tests/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf.gz'
    bed1 = '/gpfs/gpfs1/home/jholt/sanger_less_tests/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed'
    
    filterBedDir = '/gpfs/gpfs1/home/jholt/sanger_less_tests/data/bed_empty_filtered'
    if not os.path.exists(filterBedDir):
        os.makedirs(filterBedDir)

    for x in range(1, 8):
        
        #print(f'/gpfs/gpfs1/home/jholt/sanger_less_tests/data/HG00{x}*GRCh38*.vcf.gz')
        candVCF = glob.glob(f'/gpfs/gpfs1/home/jholt/sanger_less_tests/data/HG00{x}*GRCh38*.vcf.gz')
        if x == 1:
            candVCF = [vcf1]
        assert(len(candVCF) == 1)
        candBED = glob.glob(f'/gpfs/gpfs1/home/jholt/sanger_less_tests/data/HG00{x}*GRCh38*.bed')
        if x == 1:
            candBED = [bed1]
        assert(len(candBED) == 1)

        newBedFN = f'{filterBedDir}/HG00{x}_GRCh38_filtered.bed'
        findEmptyRegions(candVCF[0], candBED[0], newBedFN)