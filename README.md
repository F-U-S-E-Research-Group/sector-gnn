# Sector Graph Neural Network for Financial Markets

This repository contains the code and experiments for a research project investigating whether explicitly modeling cross-sector relationships using graph neural networks (GNNs) improves short-horizon return direction prediction in financial markets.

This project is developed by the **F.U.S.E Research Group (Financial Understanding via Statistical Exploration)** as a student-led research effort.

---

## Project Motivation

Financial sectors exhibit strong interdependencies driven by macroeconomic factors, supply chains, and investor behavior. Despite this, many predictive models treat sectors independently and rely solely on node-level features such as recent returns or volatility.

Graph neural networks provide a natural framework for incorporating relational structure into predictive models. This project evaluates whether modeling sector relationships via a dynamic correlation graph improves out-of-sample predictive performance compared to non-graph baselines.

---

## Task Description

- **Prediction task:** Binary classification of five-day forward return direction (up/down)
- **Universe:** Eleven major U.S. sector exchange-traded funds (ETFs)
- **Evaluation methodology:** Strict chronological train–validation split to prevent lookahead bias

---

## Models Implemented

| Model | Description |
|------|------------|
| Always-Up Baseline | Predicts positive returns for all sectors |
| Feature-Only MLP | Multilayer perceptron using node-level features only |
| Graph Neural Network | Graph convolutional network using sector correlation structure |

---

## Key Results (Chronological Split)

| Model | Validation Accuracy |
|------|---------------------|
| Always-Up Baseline | ~0.56 |
| Feature-Only MLP | 0.542 |
| Graph Neural Network | **0.605** |

The feature-only model fails to outperform the trivial baseline under chronological evaluation, while the graph neural network achieves a substantial improvement, indicating that cross-sector relational structure provides meaningful predictive signal.

---

## Data and Features

- **Data source:** Yahoo Finance  
- **Time period:** 2015–2024  

**Node-level features:**
- Five-day return  
- Ten-day return  
- Five-day rolling volatility  
- Five-day average trading volume  
- Distance from ten-day moving average  

All features are standardized per sector over time.

---

## Graph Construction

For each trading day, a dynamic sector graph is constructed using a rolling 60-day correlation matrix of sector returns. Edges are created when the absolute correlation between two sectors exceeds a fixed threshold, with edge weights corresponding to correlation values.

This allows the graph structure to evolve over time in response to changing market relationships.

---

## Repository Structure

sector-gnn/
├── src/
│ ├── download_data.py
│ ├── features.py
│ ├── labels.py
│ ├── dataset.py
│ ├── splits.py
│ ├── baselines.py
│ ├── mlp.py
│ ├── gnn.py
│ ├── train_mlp.py
│ └── train_gnn.py
├── data/
├── results/
├── README.md
└── requirements.txt



---

## Usage

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
Train models:


python src/train_mlp.py
python src/train_gnn.py
Paper
A full academic preprint accompanies this repository and describes the methodology, experiments, and results in detail. The code is designed to support reproducibility and transparent evaluation.

## Disclaimer
This project is for educational and research purposes only and does not constitute financial advice.
