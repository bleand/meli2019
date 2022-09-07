MeLi Data Challenge 2019
=======


This repo contains my solution to the MeLi Data Challenge hosted by [Mercado Libre](https://www.mercadolibre.com) on 
2019

It's a formatted version to make it easier to follow, and to run on newer versions of the libraries, but nothing has
been changed from the original design.

Data
----
Data can be downloaded from the following links:
 * [Train.csv.gz](https://meli-data-challenge.s3.amazonaws.com/train.csv.gz)
 * [Test.csv](https://meli-data-challenge.s3.amazonaws.com/test.csv)

Notebooks
---------
The notebooks contain a detailed step by step of the solution. 

Only `1%` of the data is used in the notebooks to make it easier for the user to 
read through and to be able to execute it in a reasonable time. The entire dataset contains 20M rows
and it would take hours to run.

Notebook should be run in numerical order:

**1 - EDA.ipynb**:   
* EDA steps 
* creation of a 1% sample of spanish data. 

**2 - PreProcess.ipynb**:   
* Steps needed to clean the data 
* Creation of the embedding representations
* Split into train and test.

**3 - Train Model.ipynb**:   
* Binarized labels
* Evaluation dataset extracted from the train data
* Class weight calculation
* Model training
* BACC calculation on test set

Scripts
-------

The scripted version is designed to train on the entire dataset. One model per language is created.

It will also save the data in multiple steps to make a re-run easier.

To execute the pipeline, run `main.py` from `scripts` with the desired flags:

| flag | description |
|----- | ----------- |
| --clean-tset | Loads raw data from `RAW_TRAIN` and starts from scratch. Useful when title cleaning changed. |
| --create-tset | Re-creates the training set pickles. Useful when pre-processing changed. |
| --train | Trains the models. If not present, loads the model un runs prediction on test data set. |
| --calc-labels | Re-calculates one-hot indices. Useful when data structure changed (i.e., subsample). |
| --fresh-start | Creates a new model from scratch. If not present, loads the model and continues training. |
| --sample | Runs in `1%` of the data. Useful to test the code  |
| --multi-gpu | Runs `tensorflow` as a parallel model using 2 GPUs |

In order to test the code on a fresh start, run the following line.
```commandline
python .\main.py --clean-tset --create-tset --train --calc-labels --fresh-start --sample
```
