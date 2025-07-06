# Climate and Lassa Fever Forecasting Project

This project explores and compares multiple forecasting approaches (MAR and LSTM models) applied to Lassa fever incidence and climate data.

## Structure

- **data/**: Raw and processed datasets
- **models/**: MAR and LSTM implementations
- **notebooks/**: Jupyter notebooks for training, evaluation, and explainability
- **utils/**: Shared utilities for preprocessing, plotting, etc.
- **results/**: Visuals, saved models, and metrics

## Model Variants

- **MAR Models**:
  - `MAR_model_per_state/`
  - `MAR_model_per_state_one_output/`

- **LSTM Models**:
  - `LSTM_model_per_state/`
  - `LSTM_model_per_state_one_output/`

## Getting Started

1. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    ```

2. Install dependencies:
    ```bash
    uv pip install --editable .
    ```

3. Launch notebooks:
    ```bash
    jupyter notebook
    ```

## Notes

- All figures and SHAP outputs are stored under `results/`.
- Raw climate and epidemiological data are located in `data/raw/`.
