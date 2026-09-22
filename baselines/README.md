# SOUND
## Description
To optimize efforts in Software Quality
 Assurance (SQA) activities, we propose a novel Code-line-level defect prediction (CLDP) method,
 Spectrum infOrmation and caUsality aNalysis based coDe-line
level defect prediction (SOUND).

This is the source code for the paper "Boosting Code-line-level Defect Prediction with
 Spectrum Information and Causality Analysis".

## Usage
### SOUND
SOUND consists of four parts: `preprocessing`, `line-level analysis`, `file-level classification`, and `global ranking`.

The commands below should be executed under the folder [`/sound/src/`](/sound/src/)
#### Preprocessing and Line-level Analysis
1. run the command to get Bag of Tokens (BoTs) of files and the spectrum information of each token

```shell
python extract_new_csv.py
```

2. run the command to evaluate the suspicion score for each token

```shell
python score.py
```

3. run the command to select the BoTs of top100 tokens
```shell
python select_top.py
```

4. run the command to learn the causal graph

```shell
python run_discovery.py
```

#### File-level Classfication and Global Ranking
1. run the [`main.py`](/sound/src/main.py), combining the `file-level classfication` and `global ranking`
```shell
python main.py
```
### Baselines
We provide the source code of SOTA baselines:
 | Baseline|Source path|Description|
 |---|---|---|
 |GLANCE|[`/sound/src/models/glance.py`](/sound/src/models/glance.py) | An approach that incorporates control elements and statement complexity. There are three variants based on the type of file-level classifier: GLANCE-LR, GLANCE-MD, and GLANCE-EA.|
 |LineDP|[`/sound/src/models/linedp.py`](/sound/src/models/linedp.py)|An MIT-based approach that uses a model-agnostic technique, LIME, to explain the filelevel classifier, extracting the most suspicious tokens.|
 |DeepLineDP|[`/sound/src/models/DeepLineDP_model.py`](/sound/src/models/DeepLineDP_model.py)| A deep learning approach to automatically learn the semantic properties of the surrounding tokens and lines to identify defective files/lines.|
 |N-gram|[`/sound/src/models/Ngram/n_gram.java`](/sound/src/models/Ngram/n_gram.java)| A typical natural language processing-based approach, using the entropy value to infer the naturalness of each code token.|
 |ErrorProne|[`/sound/src/models/ErrorProne/run_ErrorProne.py`](/sound/src/models/ErrorProne/run_ErrorProne.py)|A Google’s static analysis tool that builds on top of a primary Java compiler (javac) to check errors in source code based on a set of error-prone rules.|

 #### GLANCE and LineDP
 1. run the command in [`/sound/src/`](/sound/src/) to get results predicted by `GLANCE` and `LineDP` , details can be edited in `def run_default()` function
 ```shell
 python main.py
 ```

 #### DeepLineDP
 1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to modify the format of the dataset, replacing " with "" to accommodate the requirements of DeepLineDP's code and generalize the dataset in both `File-level` and `Line-level`

 ```shell
python preprocess_file_dataset.py
python preprocess_line_dataset.py
 ```

2. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for file-level model training

```shell
python preprocess_data.py
```

3. run the command in [`/sound/src/models/`](/sound/src/models/) to train Word2Vec models

```shell
python train_word2vec.py
```

4. run the command in [`/sound/src/models/`](/sound/src/models/) to train DeepLineDP models

```shell
python train_model.py
```

5. run the command in [`/sound/src/`](/sound/src/) to make a prediction of each software release

```shell
python generate_prediction.py
```

6. run the command in [`/sound/src/`](/sound/src/) to get the result of experiments (after getting the N-gram and ErrorProne results)

```shell
python get_DeepLineDP_Ngram_ErrorProne_evaluation.py
```

#### N-gram
1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for `N-gram` and `ErrorProne`

```
python export_data_for_line_level_baseline.py
```

2. run the command in [`/sound/src/models/Ngram/`](/sound/src/models/Ngram/) to obtain results from `N-gram`

```shell
javac -encoding UTF-8 -cp ".;slp-core.jar;commons-io-2.8.0.jar" n_gram.java
java -cp ".;slp-core.jar;commons-io-2.8.0.jar" n_gram
```

#### ErrorProne
1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for `N-gram` and `ErrorProne`

```
python export_data_for_line_level_baseline.py
```
2. run the command in [`/sound/src/models/ErrorProne/`](/sound/src/models/ErrorProne/) to get rsults from `ErrorProne`

```shell
python run_ErrorProne.py
```
### RQs
The data and figures for RQs can be gennerated by files in [`/sound/exp/`](/sound/exp/)
