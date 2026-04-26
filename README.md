# Time-Series Forecasting: Comparative Analysis (MLP, GRU, LSTM, Transformer)

A comprehensive study of different neural network architectures for time-series forecasting. This project compares traditional dense layers with recurrent and attention-based models using dynamic hyperparameters.

---

## 📺 Project Presentation
For a detailed walkthrough of the parameters, architecture logic, and ablation results, watch the explanation video here:
[**Watch on YouTube: Project Overview & Code Walkthrough**](https://youtu.be/7Dj8Es7gO7Y?si=rtaGCi9nI1jfGFDJ)

---

## 🚀 Overview
This repository implements a time-series forecasting pipeline using **PyTorch**. The project evaluates four distinct models on two datasets (AirPassengers and Electric Production) to determine how well they capture trends and seasonality.

### 🧠 Architectures
- **MLP:** Baseline multi-layer perceptron.
- **Custom GRU:** A from-scratch implementation of Gated Recurrent Units using the update and reset gate equations.
- **LSTM:** Long Short-Term Memory network utilizing `nn.LSTM`.
- **Transformer:** A modern attention-based approach using `nn.TransformerEncoder`.

---

## 🔢 Dynamic Hyperparameter Derivation
The model configuration is uniquely derived from a student **Roll Number (102317237)** to ensure personalized testing constraints:

| Parameter | Formula | Calculated Value |
| :--- | :--- | :--- |
| **Window Size** | `(sum(digits) % 10) + 8` | **14** |
| **Prediction Horizon** | `(roll[-2:] % 3) + 1` | **1** |
| **Hidden Size** | `(roll[:3] % 16) + 8` | **13** |

---

## 🛠️ Project Structure
- **Data Preprocessing:** Z-score normalization and sliding window generation.
- **Sliding Window:** Converts 1D sequences into supervised learning pairs $(X, y)$.
- **Ablation Study:** Conducted on the Electricity dataset to analyze the impact of window size (Half vs. Original vs. Double).
- **Metrics:** Evaluation using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

---

## 📈 Key Observations
- **Sequence Memory:** The Custom GRU and LSTM consistently outperform the MLP by maintaining a hidden state that captures temporal dependencies.
- **Window Impact:** The ablation study highlights that a window size that is too small leads to underfitting, whereas a window too large can introduce noise, as seen in the Electricity dataset.
- **Error Trends:** Models tend to struggle with extreme peak values in the electricity data, likely due to the MSE loss function favoring "safer" mean-centric predictions.

---

## 💻 Requirements
- `numpy`
- `pandas`
- `torch`
- `matplotlib`
- `scikit-learn`

---

## 📄 License
This project was developed as part of a Deep Learning assignment at Thapar Institute of Engineering and Technology.
