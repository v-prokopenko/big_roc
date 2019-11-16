# Big ROC
Analysis of feature space for a large numbers of subjects or 
what to do if all similarity scores don't fit in-memory.
##### Note: from now on by ROC curve we will mean FNR (1 - TPR) as a function of FPR (instead of a conventional TPR as a function of FPR)
# Getting started
To setup the python environment you can use either create conda environment using .yml file (`conda env create -f environment.yml`) 
or install packages using pip (`pip install -r requirements.txt`).
# Usage
## Run using a configuration file
You can run the program using the configuration .json file:
```console
python __main__.py --config-file=config.json
```
Configuration .json file has the following format:
```
{
  NOTE (not a json field): duplicate data_files are not removed. So if you specify the same file 2 times, the program will run twice for it
  "data_files": list of paths to the files with features (accepts Unix style pathname patterns),
  "n_features": list of numbers of features to subsample from the given files with data,
  "n_subjects": list of numbers of subjects to subsample from the given files with data,
  "n_repetitions": integer value that specifies the number of times to repeat the analysis for given data file, number of features and subjects,
  "repetition_start"[Optional, see --merge option]: index to start repetition count from (if not specified it's 1),
  "output_dir": path to the directory to where the will be saved
}
```
For example:
```json
{
  "data_files": ["data/synthetic_data/Band*.csv"],
  "n_features": [10],
  "n_subjects": [1000, 2000],
  "n_repetitions": 3,
  "repetition_start": 11,
  "output_dir": "TestAnalysis5"
}
```
It will create n_repetitions * (number of data files (files from the expanded data_files list)) * 
(number of items in n_features list) * (number of items in n_subjects list) folders at output_dir folder. 
So if in synthetic data folder we have 4 files: Band3.csv, Band4.csv, Band8.csv, Band9.csv, 
then number of folders will be 3 * 4 * 1 * 2 = 24.

Each folder will contain 3 files:
1) Genuine and Impostor histograms with 1000 points.
2) ROC curve with 1000 points uniformly spaced over the FPR axis.
3) ROC metrics (like EER) and Genuine and Impostor distribution statistics (like median or IQR)

Then after all computations are finished it will collect all metric values into 1 file at output_dir location (currently called "ManyMetricsLargeScaleAnalysis.csv").
## Merge the results
In case you have multiple results directories with possibly colliding directories you can run the script in the "merge" mode.
You can do it by running the command like this:
```console
python __main__.py resuts1 results2 other_results* output_dir --merge
```
It will go over all specified results directories and copy each "run folder" into output_dir. 
It will save runs from all specified results directories starting with repetition 1. 
In other words, it resolves the repetition collision but loses the original repetition index. 
In the end it'll collect the results for the output_folder.

Note that this command accepts Unix style pathname patterns for results directories. 
It filters the duplicate results directories.
## Collect the results
The following command does exactly the same thing that is done in the end of the run that uses a configuration file.
```console
python __main__.py --collect=FolderWithResults 
```
You can use it if for some reason the configuration file based run didn't finish or 
if you need to combine the results from different runs.
## Run the analysis without feature or subject subsampling
If you just need to compute the genuine and impostor distribution, ROC curve and metrics without any feature or subject 
subsampling you can simply run the program in a following way:
```console
python __main__.py Band3.csv Band6.csv Band9.csv OutputFolder
```

Also you can use the project directory itself (or its .zip archive) to run the script in the similar manner. For example:
```console
python big_roc.zip --config-file=config.json
```