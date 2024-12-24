# Lassa Fever Forecasting with Autoregressive LSTM Model

This repository provides the implementation of an autoregressive model to predict Lassa fever case counts and related variables such as temperature, humidity, and precipitation. The model employs a Long Short-Term Memory (LSTM) recurrent neural network for robust time-series forecasting.

## Model Overview

The model follows an autoregressive formulation:

\[
x_t = f(x_{t-1})
\]

where:  
- \( x_t \): Vector of observed variables (e.g., Lassa fever case counts) at time \( t \).  
- \( f \): Non-linear vector-valued function implemented as a many-to-many LSTM network.

### Key Features
- **Primary Goal:** Forecast the number of Lassa fever cases.  
- **Secondary Benefits:** Predict environmental variables contributing to disease dynamics.  
- **Timeframe:** Weekly temporal resolution.  

---

## Model Architecture

The network architecture consists of:  
1. **Three Bidirectional LSTM Layers:** Process sequential data in both forward and backward directions.  
2. **ReLU Activation Function:** Enforces non-negativity and introduces non-linearities.  
3. **Linear Transformation Layer:** Maps LSTM outputs to the final prediction space.  

The overall architecture is represented as:  
\[
x_t = f_{\text{linear}} \circ r \circ f_{\text{LSTM3}} \circ f_{\text{LSTM2}} \circ f_{\text{LSTM1}}(x_{t-1})
\]

Each LSTM unit \( \text{LSTM}_i \) has:  
- **Hidden state**: \( h_t^{(i)} \)  
- **Cell state**: \( c_t^{(i)} \)  
- Learnable parameters:  
  - Weight matrix: \( W^{(i)} \)  
  - Recurrent weight matrix: \( U^{(i)} \)  
  - Bias vector: \( b^{(i)} \)  

### LSTM Computations

For each time step \( t \), the LSTM performs the following operations:

1. **Forget Gate:**  
   \[
   f_t = \sigma(W_f^{(i)}x_t + U_f^{(i)}h_{t-1} + b_f^{(i)})
   \]

2. **Input Gate:**  
   \[
   i_t = \sigma(W_i^{(i)}x_t + U_i^{(i)}h_{t-1} + b_i^{(i)})
   \]

3. **Candidate Cell State:**  
   \[
   \tilde{c}_t = \tanh(W_c^{(i)}x_t + U_c^{(i)}h_{t-1} + b_c^{(i)})
   \]

4. **Cell State Update:**  
   \[
   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
   \]

5. **Output Gate:**  
   \[
   o_t = \sigma(W_o^{(i)}x_t + U_o^{(i)}h_{t-1} + b_o^{(i)})
   \]

6. **Hidden State:**  
   \[
   h_t = o_t \odot \tanh(c_t)
   \]

---

### Model Parameters

- **Neurons per LSTM unit:** 30  
- **Lookback period:** 4 weeks  
- **Dropout:** 30%  
- **Input dimension:** 7  

Parameter dimensions:  
- Weight matrix: \( W^{(i)} \in \mathbb{R}^{30 \times 7} \)  
- Recurrent weight matrix: \( U^{(i)} \in \mathbb{R}^{30 \times 7} \)  
- Bias vector: \( b^{(i)} \in \mathbb{R}^{30} \)  

---

## Data and Training

- **Dataset:** Weekly Lassa fever surveillance data (2018–2023) from Bauchi, Edo, and Ondo States, Nigeria.  
- **Training Period:** 2018–2022  
- **Testing Period:** 2023  

### Loss Function  

The loss function is defined as:  
\[
L(W, b) = \mathbb{E}\big[(f(x_{t-1}) - x_t)^2\big] + \lambda \mathbb{E}\big[\max(0, -x_t)\big]
\]

where \( \lambda = 0.6 \).  

### Optimization and Hyperparameters  
- **Optimizer:** ADAM  
- **Epochs:** 2000  
- **Batch size:** 32  

---

## Explainability with SHAP

The model’s predictions are interpreted using SHapley Additive exPlanations (SHAP). The SHAP value for a feature \( i \) is computed as:  
\[
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \big[f(S \cup \{i\}) - f(S)\big]
\]

where:  
- \( F \): Set of all features.  
- \( S \): Subset of features excluding \( i \).  
- \( f(S) \): Model prediction using features in \( S \).  

---

## Implementation Details

- **Programming Language:** Python  
- **Framework:** PyTorch  

---

## Visualizations  

Figures include:  
- General architecture of the model.  
- Unrolled architecture across \( T \) time steps.  

---

## Repository Contents

- `model.py`: Core LSTM model implementation.  
- `training.py`: Training script with data preprocessing and SHAP analysis.  
- `data/`: Sample dataset.  
- `results/`: Visualizations and prediction outputs.  

---

## How to Use  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/lassa-fever-lstm.git
   ```  
2. Create a Python or Conda virtual environment.  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Train the model:  
   ```bash
   python training.py
   ```  

---

## Citation  

If you use this work, please cite:  

Rebekah L. G, Charles I.S, Yaknan J.G, Dominik B., Sabine D., *Towards an Integrated Surveillance for Lassa Fever: Evidence from the Predictive Modeling of Lassa Fever Incidence in Nigeria*, Journal Title, 2025  

---
