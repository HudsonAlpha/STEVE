# Systematic Training and Evaluation of Variant Evidence (STEVE)
The primary purpose of this project is to reduce the number of Sanger sequencing requests that are required by the core Genome Sequencing pipeline.

## Overview
The core idea is to use the Genome in a Bottle Truth Set(s) to gather a collection of high confidence variants that can be used to train a model to detect when a variant is likely a true positive (i.e. a real variant) or a false positive. This repository contains scripts necessary to perform the steps to create and then use these models. The following summarizes the steps:

1. Feature extraction - translate variant-level statistics into numerical features
2. Model training - trains multiple models using the extracted features and reports multiple summary statistics on performance
3. Variant evaluation - runs the models and predicts whether a variant is a false positive or true positive

## Snakemake Pipeline
The easiest way to invoke this pipeline is to use the provided conda environment and the snakemake wrapper script, `RunTrainingPipeline.py`. This will first require some pipeline configuration, sample configuration, and then a final run step.

### Pipeline Configuration
Inside `PipelineConfig.py` is a set of information describing where files can be found within your environment. Generally, these values need to be set only once during the initial setup. The following are key variables that should be modified:

#### Required
1. `DATA_DIRECTORY` - The full path to a directory containing pipeline outputs. The pipeline assumes that variant files are located at `{DATA_DIRECTORY}/variant_calls/{reference}/{aligner}/{caller}/{sample}.vcf.gz`.  Similarly, it assumes that RTG VCFEval outputs are located at `{DATA_DIRECTORY}/rtg_results/{reference}/{aligner}/{caller}/{sample}`.  Both of these are required for the pipeline to run successfully. NOTE: if using a variant caller that is not specified in `model_metrics.json`, additional steps may be required (see section "Feature Extraction" for details).
2. `REFERENCES`, `ALIGNERS`, `VARIANT_CALLERS` - Each of these is a list of references, aligners, or variant callers that will be used to populate the paths above (e.g. `{reference}/{aligner}/{caller}`). Every combination of the three lists will be used.
3. `FULL_PIPES` - similar to the above options, but this one allows for specific tuple triplets (i.e. instead of every combination).

#### Optional
1. `THREADS_PER_PROCS` - positive integer, the number of threads to use for training and testing
2. `LATEX_PATH` - path to a `pdflatex` binary; only required if attempt to generate a supplemental document similar to the one released with the paper
3. `ENABLE_SLACK_NOTIFICATIONS` - if True, slack notifications will be sent upon pipeline completion
4. `ENABLED_SLACK_URLS` - a JSON dictionary file where each key is a channel or label and the value is a Slack endpoint URL
5. `ENABLED_SLACK_CHANNEL` - the specific channel or label in the above JSON to send messages to

### Sample Configuration
Before running the pipeline, a sample configuration JSON is required (the one used for our paper is in `GIAB_v1.json`).  This JSON file contains a dictionary where each key is a `{sample}` name or label containing a subdictionary with metadata.  Whenever a new sample is added to the training/testing set, this file will need to be updated.

The following fields are optional metadata keys currrently used in the automatic supplemental generation script:
1. `sample` - the sample source; currently expected to be one of: `NA12878`, `HG002`, `HG003`, `HG004`, or `HG005`
2. `prep` - a string describing the preparation method of the data; e.g. "Clinical PCR" or "PacBio CCS"

### Cluster Configuration
We have currently configured the pipeline to run using an LSF cluster.  If a different cluster type is to be used (or run locally), then several modification may be necessary to the `RunTrainingPipeline.py` script itself. Note that these changes should only have to be made when switching to a different execution environment.  We attempt to describe the possible changes below:
1. `SNAKEMAKE_PROFILE` - we assume that the user has a [Snakemake profile](https://github.com/Snakemake-Profiles/doc) which has configured how/where to run snakemake jobs.  Once created, the value of this variable is the string (e.g. `"lsf"`) describing the profile.
2. If running locally, configuration changes to the snakemake command itself may be necessary. These are located in variable `snakemakeFrags`, contact the repo owner or submit a ticket for assistance.

### Training Configuration (Optional)
There are several options available that adjust way training of models is performed (e.g. models used, hyperparameters, training method, etc.).
These options are available in `TrainingConfig.py` and generally described in-line with the parameter.
However, some are of critical importance and should be considered for training:

1. `ENABLE_AUTO_TARGET` - If set to `True`, this option enables an alternate method for determining the target recall that is automatically calculated based on the observed precision and the desired global precision (another parameter `GLOBAL_AUTO_TARGET_PRECISION`). *Global precision* in this context is the combined precision of the upstream pipeline followed by STEVE identifying false positive calls from that pipeline.  For example, if the desired global precision is 99% and the upstream pipeline achieves 98% precision by itself, then the models trained by STEVE need to capture 1% out of the 2% false positives to achieve 99% global precision. This means the target recall will be set to 0.5000 indicating that half of all false positives need to be identified to achieve the desired global precision. This approach allows for pipeline that _already_ have a high precision to have lower/easier targets in STEVE to achieve the same global precision. In practice, this allows you to swap the upstream pipelines without needing to recalculate/adjust the target/accepted recall values to account for a more or less precise upstream pipeline.
2. `ENABLE_FEATURE_SELECTION` - If set to `True`, this option enabled feature selection prior to model training. Numerous parameters adjust how exactly the feature selection is performed. In general, enabling feature selection leads to longer training times, but may improve the results by removing unnecessary and/or redundant features using a systematic approach.

### Setting up conda environment
Assuming conda or miniconda is installed, use the following two commands to create and then activate a conda environment for the pipeline:

```
conda env create -f scripts/conda_config.yaml
conda activate steve
```

### Running the Pipeline
Assuming the above configuration steps were successful, all that remains is to run the pipeline itself.  Here is an example execution of the pipeline used in the paper:

```
python3 scripts/RunTrainingPipeline.py -dest -x ./sample_metadata/GIAB_v1.json
```

For details on each option, run `python3 scripts/RunTrainingPipeline.py -h`.  The following sections describe what the pipeline is actually doing.

## Sample Stats (Optional)
This optional step will gather true and false positive call counts from the feature and format them into a tab-delimited output format.  It will look through a given features folder and identify all available samples automatically.  The script `scripts/PrintDataReport.py` performs these steps and is automatically invoked with the `-d` option of `scripts/RunTrainingPipeline.py`. 

## Feature Extraction
The first main step is to extract features from the true positive and false positive variant calls. Currently, any new "pipeline" (consisting of an aligner and a variant caller) will require its own set of features. This is because each variant caller produces its own set of metrics, many of which are not shared between callers. Additionally, the method by which the metrics are calculated may vary from caller to caller. The scripts in this pipeline assume that the full "pipeline" has already been run and the outputs are in a standardized location. Whenever a new variant caller is used, there may be multiple updates necessary to handle the caller.  Here is the summary of where to check:

1. Check `scripts/model_metrics.json` to determine whether a newly-defined model should be made or an old one copied.
2. If any new statistics need to be copied, they should be added to the appropriate `CALL` or `INFO` sections in the file.  Updates to the `scripts/ExtractFeatures.py` may be necessary.
3. If any new `MUNGED` statistics are added, the code within `scripts/ExtractFeatures.py`, function `getVariantFeatures(...)` will need to be modified to handle a custom-processed feature.

## Model Training
Once the features are extracted for a model, the data is split at a 50:50 ratio in training and testing datasets while preserving sample labels.  The training data is then used for a leave-one-sample-out cross-validation (CV) that also performs hyperparameter selection simultaneously. After CV is complete, the final model is trained on the entire training set and evaluated on the entire testing set. The script `scripts/TrainModels.py` performs all these steps and is automatically invoked with the `-t` option of `scripts/RunTrainingPipeline.py`.

## Model Evaluation (Optional)
This optional step will gather feature importance results from the trained clinical models (a minimum and target recall must be specified).  The script `scripts/ExtractELI5Results.py` performs these steps and is automatically invoked with the `-e` option of `scripts/RunTrainingPipeline.py`. 

## Summarized Results (Optional)
This optional step will gather summary results for the final models and print the results in a tab-delimited format.  It includes the best results for each model type evaluated and a summary at the bottom indicating which models were selected for a set of clinical criteria.  The script `scripts/PrintModelReport.py` performs these steps and is automatically invoked for our clinical criteria and a strict criteria with the `-s` option of `scripts/RunTrainingPipeline.py`. 

## Variant Evaluation
The following command will actually use the trained models to evaluate variant calls present in a VCF file:

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
1. `-m [model]` - the model can be "clinical", "all", or a specific model name (see "all" for the options); "clinical" will use a model with defined clinical criteria inside the script; "all" will use every model available; "best" will select the model with the lowest final FPR at the selected evaluation threshold (NOTE: this may be undesirable for variant types with low training/testing volume)
2. `-r [recall]` - for "clinical", this is the target recall (minimum is defined inside the script currently); for other `[model]` parameters, this is the evaluation recall
3. `-c [codicem]` - a Codicem Sanger download file with variants to evaluate
4. `-v [coordinates]` - comma separated, coordinates list of the form "chr1:1234-1234" for the variants; all variants within a given range will be evaluated
5. `[model_directory]` - the directory containing the models file (this will be under the pipeline subfolder)
5. `[vcf_filename]` - the raw VCF file, make sure it is the same format as the selected model

## License
For non-commerical use, STEVE is released under the Apache-2.0 License. For commercial use, please contact [jholt@hudsonalpha.org](jholt@hudsonalpha.org).

## References
Published paper:

[Holt, James M., et al. "Reducing Sanger confirmation testing through false positive prediction algorithms." _Genetics in Medicine_ (2021): 1-8.](https://www.nature.com/articles/s41436-021-01148-3)

Original pre-print on bioRxiv:

Holt, J. M., Wilk, M., Sundlof, B., Nakouzi, G., Bick, D., & Lyon, E. (2020). Reducing Sanger Confirmation Testing through False Positive Prediction Algorithms. _bioRxiv_.
