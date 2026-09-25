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
```
## FAQ's

**Q: Why use a Vision Transformer instead of just feeding numerical price data into an LSTM?**
**A:** Human traders frequently look at chart topography—support/resistance levels, trendline bounces, and candlestick wicks—which are highly spatial. While an LSTM is excellent at processing sequences of numbers, ViTs excel at recognizing spatial geometries across a canvas. Project Apex tests if capturing both numerical and spatial data yields an informational edge.

**Q: Can I run this live with real money?**
**A:** No. This is a research platform. Passing data from a broker into the prediction engine is technically trivial, but the backtester does not account for order-book depth, latency, or API rate limits. High classification accuracy does not guarantee a profitable equity curve.

**Q: Why is the model predicting NEUTRAL so often?**
**A:** By design. Markets spend roughly 70% of their time in choppy, non-directional consolidation. The dynamic volatility threshold (e.g., $0.5 \times ATR$) ensures the model only predicts UP or DOWN when the anticipated move is larger than standard market noise.

**Q: Does the model need a GPU to run?**
**A:** For training, an NVIDIA GPU (CUDA) or Apple Silicon (MPS) is strongly required to handle the Vision Transformer and image datasets. For inference (running the Streamlit dashboard to get predictions on a single stock), a standard CPU is perfectly sufficient.

**Q: How is Project Apex different from standard technical indicators like RSI, MACD, or Moving Averages?**

**A:** Standard technical indicators are fundamentally static mathematical heuristics, whereas Project Apex is a dynamic, multimodal machine learning engine. The differences break down into four key categories:

1. **Deep Neural Learning vs. Fixed Heuristics**
   * *Standard Indicators:* Rely on rigid, hardcoded formulas invented decades ago (e.g., RSI calculates average gains vs. losses over a fixed $N$ periods). They follow the exact same logic regardless of macroeconomic context or changing market regimes.
   * *Project Apex:* Uses a Vision Transformer with millions of parameters to automatically *learn* complex, non-linear relationships. It adapts to subtle multi-candle dynamics that human analysts see but simple math equations miss.

2. **Multimodal Spatial Context vs. Isolated Metrics**
   * *Standard Indicators:* Measure only one specific metric at a time (e.g., momentum, volatility, or trend direction) in complete isolation from the rest of the chart.
   * *Project Apex:* Mimics how a human quantitative trader actually looks at a screen. It fuses **spatial visual context** (how candlestick shapes, wicks, bodies, and support/resistance geometries *look* visually) with an array of **numerical indicators** simultaneously.

3. **Probabilistic Forecasting vs. Static Thresholds**
   * *Standard Indicators:* Provide binary or static states without statistical confidence (e.g., "RSI > 70 is overbought"). They cannot quantify the probability of a successful trade.
   * *Project Apex:* Outputs a calibrated probability distribution across discrete classes (e.g., $P(\text{UP})=73.4\%$, $P(\text{DOWN})=18.1\%$, $P(\text{NEUTRAL})=8.5\%$). This allows for dynamic position sizing based on statistical confidence rather than arbitrary threshold crossings.

4. **Target-Optimized vs. Lagging**
   * *Standard Indicators:* Inherently lagging. They summarize what the price *has already done* over a past window without explicitly optimizing for a specific future target.
   * *Project Apex:* Specifically trained via Backpropagation to minimize loss against a **defined forward horizon** (e.g., "Where will the price be exactly 3 trading days from right now?"). It is explicitly forward-looking rather than backward-summarizing.
