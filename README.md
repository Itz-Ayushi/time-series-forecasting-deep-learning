# UCS761 — Sequence Modeling Assignment

A time-series forecasting pipeline built from scratch for the UCS761 lab assignment at Thapar Institute of Engineering & Technology. Implements and compares MLP, Custom GRU, LSTM, and Transformer models on two real-world datasets.

---

## Roll Number Derived Parameters

Roll Number: `102317237`

| Parameter | Formula | Value |
|---|---|---|
| `window_size` | (sum of digits) % 10 + 8 | **17** |
| `prediction_horizon` | (last 2 digits) % 3 + 1 | **2** |
| `hidden_size` | (first 3 digits) % 16 + 8 | **18** |

> Last digit is **7 (ODD)** → Model assignment: **Custom GRU** (implemented from scratch)

---

## Datasets

| Dataset | Source |
|---|---|
| Airline Passengers | [JBrownlee GitHub](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv) |
| Electric Production | [Kaggle – Predict Electricity Consumption](https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption) |

Both datasets are z-score normalized before use.

---

## Project Structure

```
.
├── solution.py          # Main implementation file
└── Electric_Production.csv
```

---

## Models Implemented

### 1. MLP (Baseline)
A simple feedforward network with no sequence awareness. Takes a flat window of past values and predicts the next `prediction_horizon` steps. Serves as a lower-bound benchmark.

### 2. Custom GRU *(from scratch — assigned model)*
Implements the full GRU update equations manually using `nn.Linear` layers — no `nn.GRU` used:
- **Update gate (z):** controls how much of the old hidden state to keep
- **Reset gate (r):** controls how much past context to forget
- **Candidate hidden state (h̃):** new information blended in
- **Hidden state update:** `h = (1 - z) * h_prev + z * h̃`

### 3. LSTM (PyTorch built-in)
Used for comparison. Handles vanishing gradients better than plain RNNs via cell state and four gates.

### 4. Transformer (PyTorch built-in)
Used for comparison. Attention-based model with no recurrence; captures long-range dependencies in parallel.

---

## Data Pipeline

```
Raw time series
      │
      ▼
 Z-score normalization
      │
      ▼
 Sliding window creation
 [t, t+1, ..., t+W-1]  →  [t+W, ..., t+W+H-1]
      │
      ▼
 Chronological 80/20 split (no shuffling)
      │
      ▼
 PyTorch tensors → model training
```

- **Window size (W):** number of past time steps given as input
- **Prediction horizon (H):** number of future steps to predict
- No data leakage — test set is strictly the final 20% of the time series

---

## Training

- Optimizer: Adam (lr = 0.001)
- Loss: MSELoss
- Epochs: 20
- No mini-batching — full batch gradient descent

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |

---

## Ablation Study

The Custom GRU is retested with three window sizes to observe how context length affects forecasting accuracy:

| Window Size | Description |
|---|---|
| `window_size // 2` | Half context |
| `window_size` | Original (baseline) |
| `window_size * 2` | Double context |

Results show how insufficient or excessive context impacts model performance.

---

## Plots Generated

- Training loss curve (GRU)
- Predicted vs. Actual values (first 100 test samples, GRU)

---

## How to Run

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Download `Electric_Production.csv` from [Kaggle](https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption) and place it in the working directory, then:

```bash
python solution.py
```

The script will run all models on both datasets, print metrics, and display plots.

---

## Key Observations

- **MLP** ignores temporal order — treats the input window as a flat feature vector, often underperforms on datasets with strong trends or seasonality.
- **Custom GRU** maintains a hidden state across time steps, allowing it to capture short-to-medium range dependencies.
- **LSTM** generally performs similarly to GRU but with additional cell state, which can help in longer sequences.
- **Transformer** can capture long-range patterns but may overfit on small datasets with only 20 epochs.
- **Smaller window size** → model sees less context, struggles with patterns needing longer history.
- **Larger window size** → more context but also more noise; may not always improve performance.

---

## Youtube Link-

-https://youtu.be/7Dj8Es7gO7Y?si=7DpVdlNBKFPpU0Lq 
