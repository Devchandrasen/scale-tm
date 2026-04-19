# Data Setup

This repository expects the public DL4Internet OD-pair traffic-matrix CSV files in `data/raw/`.
The raw CSVs are intentionally not committed because they are large public benchmark files.

Download the files from:

https://github.com/jwwthu/DL4Internet/tree/main/TrafficMatrixPrediction/OD_pair

Place them with these exact names:

```text
data/raw/Abilene-OD_pair.csv
data/raw/CERNET-OD_pair.csv
data/raw/GEANT-OD_pair.csv
```

The benchmark scripts read only the first `sample_rows` rows specified in each command, so the experiments can be run on a laptop without loading full traces into memory.

