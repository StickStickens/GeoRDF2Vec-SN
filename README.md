# GeoRDF2Vec
This GitHub repository provides the code base for the paper **GeoRDF2vec – Learning Location-Aware Entity
Representations in Knowledge Graphs**. In the following, we will explain the outline of our coding base together with the necessary parameters to run the experiments.

## Environment
The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12. To create the environment from the lockfile, run:
```
uv sync
```
This covers data preparation and analysis. The original experiment code (`modeling/`, `geo_transe/`) needs further dependencies, such as MLflow, that are not included yet.

## Exploratory analysis of dmg777k
The analysis is in [notebooks/dmg777k_eda.ipynb](notebooks/dmg777k_eda.ipynb), with its main outcomes in [reports/dmg777k_findings.md](reports/dmg777k_findings.md). To reproduce it, run:
```
uv sync
uv run scripts/download_dmg777k.py
uv run python data_preparation/dmg_data_prep.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/dmg777k_eda.ipynb
```
The last command rewrites the figures in [reports/figures](reports/figures) as PDF and PNG. Helper functions are in [src/dmg777k_eda](src/dmg777k_eda).


## Tracking of experiment runs
In order to automate the experimentation deployment as well as the logging of our experiment results we use MLflow as a tracking platform. It uses a local host together with a SQLite database where the runs are stored. The models itself are stored in a folder on a file storage called artifact. Each experiment receives a unique identifier for further processing.

## Download of datasets
For the download of the datasets the following sections elaborate on the download for both datasets with KGbench and DBPedia
### DBPedia download
In order to generate the relevant subset of DBPedia, please run first the following [shell script](data_preparation/dbpedia_download.sh) to download the data. After this, please run the following script to prepare the triple dataset for the runs
### KGBench download
To download dmg777k (397 MB, about 1 GB once unpacked) into [data/kgbench_dmg777k](data/kgbench_dmg777k), run:
```
uv run scripts/download_dmg777k.py
```
The script uses the same URL as the official [kgbench-loader](https://github.com/pbloem/kgbench-loader), verifies the archive's SHA-256 checksum and unpacks the CSV files. The data is git-ignored. More about the datasets is on the [kgbench website](http://kgbench.info).
## GeoRDF2Vec implementation
The [script](geo_rdf2vec.py) for GeoRDF2Vec can be found here which implements and runs both GeoRDF2Vec as well as normal RDF2Vec. 