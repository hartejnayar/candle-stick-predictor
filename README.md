# CANDLE STICK PREDICTOR

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

**Candle Stick Predictor** is an end-to-end financial machine learning research platform and interactive dashboard. It investigates the efficacy of combining **Vision Transformers (ViTs)**—trained on raw candlestick chart images—with traditional numerical technical indicators to predict short-horizon market direction.

Unlike typical financial ML tutorials, Project Apex adheres to rigorous quantitative research standards, strictly avoiding look-ahead bias, random data splitting, and purely theoretical metrics by integrating a robust backtesting engine and chronological validation.

---

##  Primary Research Question
> *"Can visual representations of price action learned through Vision Transformers improve short-horizon market direction prediction when compared with conventional numerical and machine-learning approaches?"*

### Secondary Objectives
1. Can a Vision Transformer reliably recognize classical mathematical candlestick patterns?
2. Does fusing visual features (chart images) with numerical features (technical indicators) outperform isolated models?
3. Does high classification accuracy translate into a positive Sharpe ratio and manageable maximum drawdown in historical backtests?

---

##  Key Features

*   **Multimodal Machine Learning:** Fuses spatial/visual data (using PyTorch ViTs) with quantitative numerical data (using MLPs) for comprehensive market analysis.
*   **Deterministic Pattern Engine:** A purely mathematical engine that identifies classical candlestick patterns (Bullish Engulfing, Morning Star, etc.) to serve as ground-truth labels.
*   **Image Generation Pipeline:** Automatically converts rolling OHLCV windows into standardized, noise-free chart images—strictly containing only information available at prediction time.
*   **Robust Time-Series Validation:** Enforces strict chronological data splits (e.g., Train: 2020-2023, Val: 2024, Test: 2025-2026) to prevent data leakage.
*   **Explainable AI (XAI):** Visualizes Vision Transformer attention maps to show *where* the model is looking on the chart when making a prediction.
*   **Interactive Streamlit Dashboard:** A professional-grade UI allowing users to select tickers, view charts, overlay ML predictions, and run on-the-fly backtests.

---

## Architecture & Data Flow

```text
[OHLCV Market Data] -> [Rolling Windows] -> [Math Pattern Engine] 
                               |
                               ├──> [Image Generation] ---> (ViT) ---> [Visual Embeddings]
                               |                                                |
                               └──> [Feature Engineering] -> (MLP) ---> [Numerical Embeddings]
                                                                                |
                                                                        [Feature Fusion]
                                                                                |
                                                                        [Classification Head]
                                                                                |
                                                                     [UP / DOWN / NEUTRAL]


candle-stick-predictor/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml               # Global parameters (tickers, timeframes, hyperparams)
│
├── data/                         # Ignored by git
│   ├── raw/                      # Raw downloaded OHLCV CSVs
│   ├── processed/                # Cleaned, normalized data
│   └── datasets/                 # Generated images and numerical feature arrays
│
├── docs/                         # Project documentation
│   ├── research.md               # Literature review and hypotheses
│   ├── architecture.md           # System design diagrams
│   └── experiments.md            # Log of ablations and results
│
├── notebooks/                    # Prototyping and EDA
│   ├── 01_data_exploration.ipynb
│   ├── 02_pattern_analysis.ipynb
│   ├── 03_dataset_generation.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_vit_training.ipynb
│   ├── 06_multimodal_model.ipynb
│   └── 07_results_analysis.ipynb
│
├── src/                          # Core source code
│   ├── data/                     # Downloaders, preprocessing, PyTorch Datasets
│   ├── patterns/                 # Mathematical pattern detection engine
│   ├── features/                 # Technical indicators (RSI, MACD, etc.)
│   ├── models/                   # PyTorch architectures (ViT, CNN, Fusion)
│   ├── training/                 # Training loops, loss functions, validation
│   ├── prediction/               # Inference engine for live data
│   ├── evaluation/               # ML metrics, ROC-AUC, attention maps
│   └── backtesting/              # Vectorized backtesting, Sharpe, drawdown calculations
│
├── models/
│   └── trained/                  # Saved .pt / .pth model weights
│
├── results/                      # Experiment outputs
│   ├── figures/                  # Charts and attention map exports
│   ├── metrics/                  # CSVs of evaluation metrics
│   └── experiments/              # Model run logs
│
├── app/                          # Streamlit UI
│   ├── app.py                    # Main dashboard entrypoint
│   └── components/               # UI modules (charts, sidebars, metrics)
│
└── tests/                        # PyTest suite
    ├── test_patterns.py
    ├── test_features.py
    ├── test_dataset.py
    └── test_prediction.py
