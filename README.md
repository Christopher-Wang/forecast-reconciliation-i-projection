# Approximate Discrete Probabilistic Forecast Reconciliation via Information Projections

This repository contains the code for **Approximate Discrete Probabilistic Forecast Reconciliation via Information Projections**.

The project introduces a novel method for reconciling probabilistic forecasts generated from regression-via-classification models in hierarchical time series. It works by projecting incoherent marginals onto the space of approximately coherent distributions.

The [TourismSmall Dataset](https://nixtlaverse.nixtla.io/datasetsforecast/hierarchical.html#tourismsmall) and accompanying cross-validation predictions are provided as an example.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gH7Zrqu8_dBYN5NPmnyDHr5HVvQSQKV5?usp=sharing)

## Setup
Create the conda environment:
```bash
conda create -n forecast_reconciliation python=3.11
conda activate forecast_reconciliation
```

Install all necessary dependencies
```shell
pip install -r requirements.txt
```

Install forecast reconciliation module
```shell
pip install -e .
```

## Reproducing Predictions
To reproduce the TabPFN-TS prediction results, run:
```shell
python -m scripts.pred_generation
```

## Reproducing Predictions
To reproduce the validation results, run:
```shell
python -m src.experiments.results_collector
```

## Running Result Dashboar
To launch the results dashboard using Streamlit, run:
```shell
streamlit run scripts/app.py
```

## 📚 Citation
If you use this code or method in your research, please cite:
```bibtex
@article{wang2025approximate,
  title={Approximate Discrete Probabilistic Forecast Reconciliation via Information Projections},
  author={Christopher Wang, Antoine Grosnit, Haitham Bou-Ammar, Jun Wang},
  year={2025},
}
