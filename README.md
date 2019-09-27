# CSL_sangerless_verification
The primary purpose of this project is to reduce the number of Sanger sequencing requests that are required by the core Genome Sequencing pipeline.

## Overview
The core idea is to use the Genome in a Bottle Truth Set(s) to gather a collection of high confidence variants that can be used to train a model to detect when a variant is likely a true positive (i.e. a real variant) or a false positive. This repository contains scripts necessary to perform the steps to create and then use these models. The following summarizes the steps:

1. Feature extraction - translate variant-level statistics into numerical features
2. Model training - trains multiple models using the extracted features and reports multiple summary statistics on performance
3. Variant evaluation - runs the models and reports whether a variant needs to be Sanger-confirmed or not

Currently, we have three NA12878 samples that are used for feature extraction and model training.

## Feature Extraction
The first step is to extract features from the true positive and false positive variant calls. Currently, any new "pipeline" (consisting of an aligner and a variant caller) will require its own set of features. This is because each variant caller produces its own set of metrics, some of which are shared and some of which are not. Additionally, the method by which the metrics are calculated may vary from caller to caller. The scripts in this pipeline assume that the full "pipeline" has already been run and the outputs are in a standardized location as produced by the [CSL Core Pipeline Analysis sub-repo](https://github.com/HudsonAlpha/CSL_validations/tree/master/core_pipeline_analysis). Whenever a new variant caller is used, there may be multiple updates necessary to handle the caller.  Here is the summary of where to check:

1. Check `scripts/model_metrics.json` to determine whether a newly-defined model should be made or an old one copied.
2. If any new statistics need to be copied, they should be added to the appropriate `CALL` or `INFO` sections in the file.
3. If any new `MUNGED` statistics are added, the code within `scripts/ExtractFeatures.py`, function `getVariantFeatures(...)` will need to be modified to handle a processed feature.

## Model Training
Once the features are extracted for a model, the next step is to perform cross-validation (i.e. leave-one-out testing) on the available datasets. After performing this testing on the whole samples, we do an 80:20 training:testing model fitting before saving the whole model. The following command will run the full training pipeline (feature extraction and model training) for all aligners and callers:

```
python3 RunTrainingPipeline.py -t -x [slids]
```

## Variant Evaluation
The following command will actually use the trained models to evaluate variant calls:

```
python3 EvaluateVariants.py \
    -m [model] \
    -r [recall] \
    -c [codicem] \
    -v [coordinates] \
    [model_directory] \
    [vcf_filename]
```

### Parameter Notes
1. `-m [model]` - the model can be "best", "all", or a specific model name (see "all" for the options); "all" will use every model available; "best" will select the model with the lowest FPR at the selected threshold (NOTE: this may be undesirable for variant types with low training/testing volume)
2. `-r [recall]` - currently accepts "0.99", "0.995", "0.999", and "1.0"
3. `-c [codicem]` - a Codicem Sanger download file
4. `-v [coordinates]` - comma separated, coordinates list of the form "chr1:1234-1234" for a single nucleotide variant
5. `[model_directory]` - the directory containing the models file (this will be under the pipeline subfolder)
5. `[vcf_filename]` - the raw VCF file, make sure it is the same format as the selected model