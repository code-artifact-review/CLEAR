# CLEAR

## Description

This repository contains the source code for the paper "Code-Line-Level Defect Prediction with Confidence Enhancement and Counterfactual Analysis".

## Datasets

The [`/dataset`](/dataset/) directory contains all 9 projects from the [Line-Level Defect Prediction dataset](https://github.com/awsm-research/line-level-defect-prediction).

The file-level datasets are stored in [`/dataset/linedp_dataset/File-level/`](/dataset/linedp_dataset/File-level/) and contain the following columns:

- `File`: The file name of a source code file
- `Bug`: A label indicating whether the source code is clean or defective
- `SRC`: The content of the source code file

The line-level datasets are stored in [`/dataset/linedp_dataset/Line-level/`](/dataset/linedp_dataset/Line-level/) and contain the following columns:

- `File`: The file name of a source code file
- `Line_number`: The line number of a defective code line
- `SRC`: The content of the source code line

## Environment

The required environment for CLEAR and the baselines is provided in the `environment.yml` file.

## Usage

### Baselines

Follow `/baselines/README.md` to obtain the results of SOUND, Glance, DeepLineDP, Ngram and ErrorProne.

After obtaining the results, the metric CSV files will be stored in the `/Result` folder.

### CLEAR

Run the following scripts in order to obtain the CLEAR results:

```bash
python 1.Data Preprocessing.py
python 2.BoT and Spectrum Extraction.py
python file_level_prediction.py
python 4.File-Level Counterfactual Analysis.py
python 5.Confidence-Enhanced Token Risk Modeling.py
python 6.Line-Level Counterfactual Analysis.py
python 7.Causal-Aware Global Ranking and Evaluation.py
python 8.Ablation without File Component.py
python 9.Ablation without Causal Tokens.py
python 10.Evaluate File Level Causal Token Effectiveness.py
```

## Experimental Results

### RQ1-RQ4

Run the following scripts to obtain the visualization results for RQ1-RQ4:

```bash
Rscript RQ1_SK-ESD.R
Rscript RQ2_file_level_components_ablation.R
Rscript RQ3_ablation_without_confidence.R
Rscript RQ4_ablation_causal_tokens.R
Rscript RQ4_file_line_causal_token_analysis.R
```

### Discussion

Run the following scripts to obtain the results reported in the Discussion section:

```bash
Rscript dis1_Recall@TopN.R
Rscript dis2_hit_over_analysis.R
```

## Additional Dataset

To use the [Additional dataset](https://github.com/Naplues/BugDet/tree/master/Dataset) adopted in the paper, uncomment `"glance_dataset"` in the Python scripts and rerun the code.

