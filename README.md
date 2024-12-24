# Lassa Fever Forecasting with Autoregressive LSTM Model

This repository provides the implementation of an autoregressive model to predict Lassa fever case counts and related variables such as temperature, humidity, and precipitation. The model employs a Long Short-Term Memory (LSTM) recurrent neural network for robust time-series forecasting.

## Model Overview

The model follows the autoregressive formulation:
\[ x_t = f(x_{t-1}) \]
where:
- \( x_t \in \mathbb{R}^n \) is the vector of observed variables (including Lassa fever case counts) at time \( t \),
- \( f \) is a non-linear vector-valued function implemented as a many-to-many LSTM network.

### Key Features
- **Primary Goal:** Forecast the number of Lassa fever cases.
- **Secondary Benefits:** Predict other environmental variables contributing to the disease dynamics.
- **Timeframe:** Weekly temporal resolution.

## Model Architecture

The network architecture is composed of:
1. **Three Bidirectional LSTM Layers**: Each layer processes sequential data in both forward and backward directions.
2. **ReLU Activation Function**: Introduces non-smoothness and enforces non-negativity in the predictions.
3. **Linear Transformation Layer**: Maps LSTM outputs to the final prediction space.

The overall architecture can be represented as:
\[ x_t = f_{\text{linear}} \circ r \circ f_{\text{LSTM3}} \circ f_{\text{LSTM2}} \circ f_{\text{LSTM1}}(x_{t-1}) \]

Each LSTM unit \( LSTM_i \) maintains:
- A **hidden state** (\( h_t^{(i)} \)),
- A **cell state** (\( c_t^{(i)} \)),
- Learnable parameters including:
  - Weight matrix \( W^{(i)} \in \mathbb{R}^{h \times d} \),
  - Recurrent weight matrix \( U^{(i)} \in \mathbb{R}^{h \times d} \),
  - Bias vector \( b^{(i)} \in \mathbb{R}^h \).

The computation for \( LSTM_i \) includes:
- Forget gate: \( f_t = \sigma(W_f^{(i)}x_t + U_f^{(i)}h_{t-1} + b_f^{(i)}) \)
- Input gate: \( i_t = \sigma(W_i^{(i)}x_t + U_i^{(i)}h_{t-1} + b_i^{(i)}) \)
- Cell update: \( \tilde{c}_t = \tanh(W_c^{(i)}x_t + U_c^{(i)}h_{t-1} + b_c^{(i)}) \)
- Cell state: \( c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \)
- Output gate: \( o_t = \sigma(W_o^{(i)}x_t + U_o^{(i)}h_{t-1} + b_o^{(i)}) \)
- Hidden state: \( h_t = o_t \odot \tanh(c_t) \)

### Model Parameters
- Number of neurons per LSTM unit: \( h = 30 \)
- Lookback period: \( T = 4 \) weeks
- Dropout: 30%
- Input dimension: \( x_t \in \mathbb{R}^7 \)

Learnable parameter dimensions:
- Weight matrix: \( W^{(i)} \in \mathbb{R}^{30 \times 7} \)
- Recurrent weight matrix: \( U^{(i)} \in \mathbb{R}^{30 \times 7} \)
- Bias vector: \( b^{(i)} \in \mathbb{R}^{30} \)

## Data and Training

- **Dataset**: Weekly Lassa fever surveillance data (2018–2023) from Bauchi, Edo, and Ondo States, Nigeria.
- **Training Period**: 2018–2022.
- **Testing Period**: 2023.
- **Loss Function**:
  \[
  L(W, b) = \mathbb{E}[(f(x_{t-1}) - x_t)^2] + \lambda \mathbb{E}[\max(0, -x_t)]
  \]
  with \( \lambda = 0.6 \).
- **Optimization**: ADAM optimizer.
- **Hyperparameters**:
  - Epochs: 2000
  - Batch size: 32

## Explainability with SHAP

To interpret the model, we used SHapley Additive exPlanations (SHAP), a game-theoretic approach to quantify the contribution of each feature. The SHAP value for a feature \( i \) is calculated as:
\[
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]
\]
where:
- \( F \): Set of all features.
- \( S \): Subset of features excluding \( i \).
- \( f(S) \): Model prediction with features in \( S \).

## Implementation Details

- **Programming Language**: Python
- **Framework**: PyTorch

## Visualizations
Figures in the paper include:
- General architecture of the model.
- Unrolled architecture across \( T \) time steps.

## Repository Contents

- `model.py`: Code for the LSTM model.
- `training.py`: Training script with data preprocessing and SHAP explainability analysis.
- `data/`: Sample dataset for demonstration purposes.
- `results/`: Visualization and prediction outputs.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lassa-fever-lstm.git
2. Create a python or conda virtual environment
3. pip install -r requirements.txt
4. python training.py
