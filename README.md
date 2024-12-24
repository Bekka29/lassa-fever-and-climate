# Lassa Fever Forecasting with Autoregressive LSTM Model

This repository provides the implementation of an autoregressive model to predict Lassa fever case counts and related variables such as temperature, humidity, and precipitation. The model employs a Long Short-Term Memory (LSTM) recurrent neural network for robust time-series forecasting.

---

## Model Overview

The model follows the autoregressive formulation:

**xₜ = f(xₜ₋₁)**

where:
- **xₜ** is the vector of observed variables (including Lassa fever case counts) at time **t**.
- **f** is a non-linear vector-valued function implemented as a many-to-many LSTM network.

---

## Model Architecture

The network architecture is composed of:
1. **Three Bidirectional LSTM Layers**: Each layer processes sequential data in both forward and backward directions.
2. **ReLU Activation Function**: Introduces non-smoothness and enforces non-negativity in the predictions.
3. **Linear Transformation Layer**: Maps LSTM outputs to the final prediction space.

The overall architecture can be represented as:

**xₜ = f_linear ∘ ReLU ∘ f_LSTM3 ∘ f_LSTM2 ∘ f_LSTM1(xₜ₋₁)**

### LSTM Computations
Each LSTM unit maintains:
- A **hidden state** (**hₜ⁽ⁱ⁾**),
- A **cell state** (**cₜ⁽ⁱ⁾**),
- Learnable parameters:
  - Weight matrix: **W⁽ⁱ⁾**,
  - Recurrent weight matrix: **U⁽ⁱ⁾**,
  - Bias vector: **b⁽ⁱ⁾**.

The computations for each LSTM unit include:
- **Forget Gate**:  
  fₜ = sigmoid(Wₚ⁽ⁱ⁾ * xₜ + Uₚ⁽ⁱ⁾ * hₜ₋₁ + bₚ⁽ⁱ⁾)

- **Input Gate**:  
  iₜ = sigmoid(Wₒ⁽ⁱ⁾ * xₜ + Uₒ⁽ⁱ⁾ * hₜ₋₁ + bₒ⁽ⁱ⁾)

- **Cell State Update**:  
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ tanh(W_c⁽ⁱ⁾ * xₜ + U_c⁽ⁱ⁾ * hₜ₋₁ + b_c⁽ⁱ⁾)

- **Output Gate**:  
  oₜ = sigmoid(Wₒ⁽ⁱ⁾ * xₜ + Uₒ⁽ⁱ⁾ * hₜ₋₁ + bₒ⁽ⁱ⁾)

- **Hidden State Update**:  
  hₜ = oₜ ⊙ tanh(cₜ)

### Model Parameters
- Neurons per LSTM unit: **30**
- Lookback period: **4 weeks**
- Dropout: **30%**
- Input dimension: **7**

Parameter dimensions:
- Weight matrix: **W⁽ⁱ⁾ ∈ ℝ³⁰×⁷**
- Recurrent weight matrix: **U⁽ⁱ⁾ ∈ ℝ³⁰×⁷**
- Bias vector: **b⁽ⁱ⁾ ∈ ℝ³⁰**

---

## Data and Training

- **Dataset**: Weekly Lassa fever surveillance data (2018–2023) from Bauchi, Edo, and Ondo States, Nigeria.
- **Training Period**: 2018–2022.
- **Testing Period**: 2023.

### Loss Function
The loss function combines mean squared error and a regularization term:

$$
L(W, b) = \mathbb{E}[(f(x_{t-1}) - x_t)^2] + \lambda \mathbb{E}[\max(0, -x_t)]
$$


where:
- **λ = 0.6**.

### Optimization
- **Optimizer**: ADAM
- **Hyperparameters**:
  - Epochs: **2000**
  - Batch size: **32**

---

## Explainability with SHAP

To interpret the model, we used SHapley Additive exPlanations (SHAP), a game-theoretic approach to quantify the contribution of each feature.

### SHAP Equation
The SHAP value for a feature **i** is calculated as:

**ϕᵢ = Σ (S ⊆ F \ {i}) [(|S|! * (|F| - |S| - 1)!) / |F|!] * [f(S ∪ {i}) - f(S)]**

where:
- **F**: Set of all features,
- **S**: Subset of features excluding **i**,
- **f(S)**: Model prediction with features in **S**.

---

## Implementation Details

- **Programming Language**: Python
- **Framework**: PyTorch

---

## Repository Contents

- `model.py`: Code for the LSTM model.
- `training.py`: Training script with data preprocessing and SHAP explainability analysis.
- `data/`: Sample dataset for demonstration purposes.
- `results/`: Visualization and prediction outputs.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lassa-fever-lstm.git
2. Create a Python or Conda virtual environment:
   conda create -n lassa-env python=3.8
   conda activate lassa-env

4. Install dependencies:
   pip install -r requirements.txt

6. Run the training script:
   python training.py
