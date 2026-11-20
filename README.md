# GeoRDF2Vec
This GitHub repository provides the code base for the paper **GeoRDF2vec – Learning Location-Aware Entity
Representations in Knowledge Graphs**. In the following, we will explain the outline of our coding base together with the necessary parameters to run the experiments.

## Repository structure
The repository is split by project stage. Each stage folder holds its own code, notebooks and results, and has a README that explains what is in it and how to run it.

| Folder | Stage | Contents |
|---|---|---|
| [original/](original) | | The authors' code, unchanged, so that the paper's results can be reproduced |
| [data/](data) | Datasets | Download and preparation scripts for dmg777k and BGS+; the data itself is git-ignored |
| [eda/](eda) | Milestone 2 | Exploratory analysis: notebooks, helper code, findings, report and figures |
| [preprocessing/](preprocessing) | Milestone 3 | Preprocessing of both datasets for the experiments: notebooks, code and a summary |
| [experiments/](experiments) | Milestone 4+ | Experiments and the evaluation plan |
| `model/` | | Output folder of the authors' training code |

Run all commands from the repository root, since the scripts, including the authors', read and write `data/` relative to it.

## Environment
The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12. To create the environment from the lockfile, run:
```
uv sync
```
This covers data preparation and analysis. The original experiment code (`original/modeling/`, `original/geo_transe/`) needs further dependencies, such as MLflow, that are not included yet.

## Original code
### GeoRDF2Vec implementation
The [script](original/geo_rdf2vec.py) for GeoRDF2Vec can be found here which implements and runs both GeoRDF2Vec as well as normal RDF2Vec.

### Tracking of experiment runs
In order to automate the experimentation deployment as well as the logging of our experiment results we use MLflow as a tracking platform. It uses a local host together with a SQLite database where the runs are stored. The models itself are stored in a folder on a file storage called artifact. Each experiment receives a unique identifier for further processing.

### DBPedia download
In order to generate the relevant subset of DBPedia, please run first the following [shell script](original/data_preparation/dbpedia_download.sh) to download the data. After this, please run the following script to prepare the triple dataset for the runs.

The authors' TransGeo and DBpedia scripts still refer to `geo_transe/` and `data_preparation/` at the repository root, as in their original layout.
