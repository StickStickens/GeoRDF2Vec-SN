# Datasets

Scripts that download and prepare our two datasets. The data itself is git-ignored. Run the commands from the repository root.

| Dataset | Task | Source |
|---|---|---|
| dmg777k | Dutch national monuments, 5 classes | [KGBench](http://kgbench.info) (Bloem et al., 2021) |
| BGS+ | British rock units, river vs glacial deposits | [mmkg](https://gitlab.com/wxwilcke/mmkg) (Wilcke et al., 2020) |

## dmg777k
```
uv run data/download_dmg777k.py
uv run python original/data_preparation/dmg_data_prep.py
```
The first script downloads the archive (397 MB, about 1 GB once unpacked) into `data/kgbench_dmg777k`. It uses the same URL as the official [kgbench-loader](https://github.com/pbloem/kgbench-loader), verifies the archive's SHA-256 checksum and unpacks the CSV files. The second is the authors' preparation script. It writes `full_triple_set.csv`, the input of `original/geo_rdf2vec.py`.

## BGS+
```
uv run data/download_bgs.py
uv run python data/bgs_data_prep.py
```
The first script downloads the files into `data/bgs` and verifies their checksums. The second is ours, because the authors' repository has no script for BGS+. It writes `full_triple_set.csv` (with each object's type, used by the preprocessing), `labels.csv` and `node_kinds.csv`, and converts the geometries from British National Grid metres to WGS84 longitude and latitude, which `original/geo_rdf2vec.py` assumes.
