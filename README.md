# Systematic Training and Evaluation of Variant Evidence (STEVE)
The primary purpose of this project is to reduce the number of Sanger sequencing requests that are required by the core Genome Sequencing pipeline.

## Overview
The core idea is to use the Genome in a Bottle Truth Set(s) to gather a collection of high confidence variants that can be used to train a model to detect when a variant is likely a true positive (i.e. a real variant) or a false positive. This repository contains scripts necessary to perform the steps to create and then use these models. The following summarizes the steps:

1. Feature extraction - translate variant-level statistics into numerical features
2. Model training - trains multiple models using the extracted features and reports multiple summary statistics on performance
3. Variant evaluation - runs the models and predicts whether a variant is a false positive or true positive

## Feature Extraction
The first step is to extract features from the true positive and false positive variant calls. Currently, any new "pipeline" (consisting of an aligner and a variant caller) will require its own set of features. This is because each variant caller produces its own set of metrics, many of which are not shared between callers. Additionally, the method by which the metrics are calculated may vary from caller to caller. The scripts in this pipeline assume that the full "pipeline" has already been run and the outputs are in a standardized location. Whenever a new variant caller is used, there may be multiple updates necessary to handle the caller.  Here is the summary of where to check:

1. Check `scripts/model_metrics.json` to determine whether a newly-defined model should be made or an old one copied.
2. If any new statistics need to be copied, they should be added to the appropriate `CALL` or `INFO` sections in the file.  Updates to the `scripts/ExtractFeatures.py` may be necessary.
3. If any new `MUNGED` statistics are added, the code within `scripts/ExtractFeatures.py`, function `getVariantFeatures(...)` will need to be modified to handle a custom-processed feature.

## Model Training
Once the features are extracted for a model, the data is split at a 50:50 ratio in training and testing datasets while preserving sample labels.  The training data is then used for a leave-one-sample-out cross-validation (CV) that also performs hyperparameter selection simultaneously. After CV is complete, the final model is trained on the entire training set and evaluated on the entire testing set. The following command will run the full training pipeline (feature extraction and model training) for all aligners and callers:

```
python3 RunTrainingPipeline.py -t -x [samples.json]
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
1. `-m [model]` - the model can be "clinical", "best", "all", or a specific model name (see "all" for the options); "clinical" will use a model with defined clinical criteria inside the script; "all" will use every model available; "best" will select the model with the lowest final FPR at the selected evaluation threshold (NOTE: this may be undesirable for variant types with low training/testing volume)
2. `-r [recall]` - currently accepts values from the "0.99", "0.995", "0.999", and "1.0"
3. `-c [codicem]` - a Codicem Sanger download file with variants to evaluate
4. `-v [coordinates]` - comma separated, coordinates list of the form "chr1:1234-1234" for a single nucleotide variant
5. `[model_directory]` - the directory containing the models file (this will be under the pipeline subfolder)
5. `[vcf_filename]` - the raw VCF file, make sure it is the same format as the selected model

## License
For non-commerical use, STEVE is released under the Apache-2.0 License. For commercial use, please contact [jholt@hudsonalpha.org](jholt@hudsonalpha.org).