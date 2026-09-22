# Baseline Metric Results

This directory stores the evaluation results of the baseline methods used in our experiments. Results are organized separately for the primary dataset (`linedp_dataset`) and the additional dataset (`glance_dataset`).

## Directory Structure

```text
metrics/
├── linedp_dataset/
│   ├── DeepLineDP.csv
│   ├── ErrorProne.csv
│   ├── GLANCE-EA.csv
│   ├── GLANCE-LR.csv
│   ├── GLANCE-MD.csv
│   ├── LineDP.csv
│   ├── Ngram.csv
│   ├── SOUND-Barinel.csv
│   ├── SOUND-Dstar.csv
│   ├── SOUND-Ochiai.csv
│   ├── SOUND-Op2.csv
│   └── SOUND-Tarantula.csv
```

## Baseline Methods

The metric files correspond to the following baseline methods:

- DeepLineDP
- ErrorProne
- GLANCE-EA
- GLANCE-LR
- GLANCE-MD
- LineDP
- N-gram
- SOUND-Barinel
- SOUND-Dstar
- SOUND-Ochiai
- SOUND-Op2
- SOUND-Tarantula

## Metrics

Each CSV file contains the evaluation results of the corresponding baseline method. The reported metrics include the main ranking-oriented and accuracy-oriented measures used in our study, such as:

- Recall@Top20%LOC
- Effort@Top20%Recall
- Initial False Alarms (IFA)
- AUC
- D2H
- FAR

