# SE4050 – Deep Learning | Lab 05

**Year 4, Semester 1 — SLIIT**

This repository contains the implementation of three deep learning tasks covering Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks, applied to sequence prediction, time-series forecasting, and sentiment analysis.

## 📁 Repository Structure

```
├── ITxxxxxxQ1.ipynb   # Task 1: Simple RNN for sequence prediction
├── ITxxxxxxQ2.ipynb   # Task 2: LSTM for stock price forecasting
├── ITxxxxxxQ3.ipynb   # Task 3: LSTM for sentiment analysis
└── README.md
```

---

## Task 1 — Simple RNN

A basic **Recurrent Neural Network (RNN)** is implemented to predict the next value in a numeric sequence. Unlike standard feedforward networks, an RNN maintains a hidden state that carries information from previous time steps, which makes it suitable for sequential data.

Experiments carried out:
- Varying the number of `units` in the `SimpleRNN` layer to observe how hidden layer size affects learning capacity.
- Tuning `epochs` and `batch_size` to optimize training performance.
- Comparing predicted vs actual values to evaluate accuracy.

---

## Task 2 — LSTM for Time-Series Forecasting

An **LSTM (Long Short-Term Memory)** network is used to forecast stock prices from historical data (`google.csv`). LSTM is a special type of RNN designed to overcome the vanishing gradient problem, allowing it to learn long-term dependencies in sequential data through its gate mechanisms (input, forget, and output gates).

### Key Concepts

**Why normalize the 'Close' prices?**
Stock prices can range across large numeric scales. Normalizing (typically to a 0–1 range using MinMaxScaler) ensures the model trains efficiently, since neural networks converge faster and more stably when input features are on a similar scale. It also prevents features with larger magnitudes from dominating the learning process.

**What is a Dropout layer?**
Dropout is a **regularization technique** used to prevent overfitting. During training, it randomly "drops" (sets to zero) a fraction of neurons in a layer on each forward pass. This forces the network to not rely too heavily on any single neuron, improving its ability to generalize to unseen data.

**Interpreting the actual vs predicted plot:**
If the predicted line closely follows the actual line, it indicates that the model has successfully learned the underlying temporal pattern in the data and is generalizing well. Large deviations suggest underfitting, overfitting, or the need for further hyperparameter tuning.

---

## Task 3 — Sentiment Analysis using LSTM

This task implements an LSTM-based text classifier to perform **sentiment analysis** on the IMDB movie reviews dataset — classifying reviews as positive or negative.

### What is Sentiment Analysis?
Sentiment analysis is a Natural Language Processing (NLP) task that determines the emotional tone or opinion expressed in a piece of text. It's widely used for analyzing customer reviews, social media posts, and feedback.

### Model Pipeline
1. **Embedding Layer** — converts words into dense vector representations (`output_dim` controls the size of these vectors).
2. **LSTM Layer(s)** — processes the sequence of word embeddings to capture contextual meaning and word order.
3. **Dropout** — added for regularization to reduce overfitting.
4. **Dense Output Layer** — produces the final sentiment classification.

### Bidirectional vs Unidirectional LSTM

| Aspect | Unidirectional LSTM | Bidirectional LSTM |
|---|---|---|
| Direction | Reads sequence left → right only | Reads sequence in both directions (forward & backward) |
| Context | Uses only past context | Uses both past and future context |
| Performance | Generally lower accuracy on tasks needing full-sentence context | Typically higher accuracy, especially for sentiment/text classification |
| Cost | Faster to train, fewer parameters | Slower to train, more parameters (roughly double) |

**Why Bidirectional LSTM often performs better for sentiment analysis:** understanding sentiment often depends on words that appear later in a sentence (e.g., negations or qualifiers). By processing text in both directions, the bidirectional model captures richer context, which typically improves accuracy and F1-score compared to a unidirectional model — at the cost of increased training time and computation.

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Jupyter Notebook / Google Colab
- Pandas, NumPy, Matplotlib

## 📌 Submission
Each notebook is renamed with the corresponding IT number (`ITxxxxxxQ1.ipynb`, `ITxxxxxxQ2.ipynb`, `ITxxxxxxQ3.ipynb`) and uploaded to this repository as per course submission guidelines.
