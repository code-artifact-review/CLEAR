## Baseline Metric Results

The baseline evaluation scripts generate the metric results used in our experiments, including:

- **Recall@Top20%LOC**
- **Effort@Top20%Recall**
- **Initial False Alarms (IFA)**
- **AUC**
- **D2H**
- **FAR**
- and other evaluation metrics.

The generated line-level evaluation results are saved under the corresponding baseline result directory. For example, the results of **Glance-MD** are stored at:

```text
Result/linedp_dataset/BASE-Glance-MD_Mixed_Sort/line_result/evaluation.csv
```

After the evaluation is completed, rename `evaluation.csv` using the corresponding baseline name and move it to:

```text
clear/exp/metrics/<dataset_name>/
```

For example, for **Glance-MD**:

```text
Result/linedp_dataset/BASE-Glance-MD_Mixed_Sort/line_result/evaluation.csv
```

should be renamed to:

```text
GLANCE-MD.csv
```

and placed at:

```text
clear/exp/metrics/linedp_dataset/GLANCE-MD.csv
```

The files under `clear/exp/metrics/` are then used for the subsequent statistical analysis and comparison with CLEAR.
