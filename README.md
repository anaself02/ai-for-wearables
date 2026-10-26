# AI for Wearables

Two projects:

- [`health dashboard/`](health%20dashboard/): Django dashboard over Google Health data.
- [`wearable-ml/`](wearable-ml/): stress-detection ML pipeline trained on WESAD.
  See [`wearable-ml/README.md`](wearable-ml/README.md) for setup and how to run it.

## Dataset

The WESAD dataset (~17 GB) is **not included** in this repo. Download it from Kaggle:

https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset

Extract it so that subject files sit at `wearable-ml/data/WESAD/S2/S2.pkl` and so on.
Full instructions are in [`wearable-ml/README.md`](wearable-ml/README.md#1-get-the-dataset).
