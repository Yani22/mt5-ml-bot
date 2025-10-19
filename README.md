# MT5-ML-Bot

## Project Overview

This project is a Python-based machine learning trading bot for the MetaTrader 5 (MT5) platform. It utilizes an ensemble of machine learning models (LightGBM, XGBoost, RandomForest, and Logistic Regression) to predict the direction of the market and execute trades accordingly. The bot is highly configurable through a `config.yaml` file, allowing for customization of data sources, feature engineering, model training, risk management, and more. It supports both live trading and backtesting.

## Core Components

*   `main.py`: The main entry point of the application. It handles the main loop, including connecting to MT5, fetching data, running the trading logic, and triggering retraining.
*   `config.yaml`: The central configuration file for the entire application.
*   `src/data_manager.py`: Manages all data-related tasks, including fetching historical data from MT5 or CSV files, caching data, and preparing data for the models.
*   `src/strategy_ml.py`: Defines the machine learning models used in the ensemble. It includes logic for training, cross-validation, and prediction.
*   `src/ensemble.py`: Handles the ensembling of the models.
*   `src/risk.py` and `src/risk_controller.py`: Manages risk, including position sizing, stop-loss/take-profit levels, and overall portfolio risk.
*   `src/execution.py`: Executes trades on the MT5 platform.
*   `trainer.py`, `tuner.py`, `backtester.py`: Scripts for training models, tuning hyperparameters with Optuna, and backtesting strategies.

## Building and Running

### Dependencies

The project's dependencies are listed in `requirements.txt`. They can be installed using:

```bash
pip install -r requirements.txt
```

### Configuration

1.  **Environment Variables:** Create a `.env` file in the root directory (you can copy `.env.example`) and add your MT5 credentials:

    ```
    MT5_LOGIN=your_login
    MT5_PASSWORD=your_password
    MT5_SERVER=your_server
    MT5_PATH=path_to_your_mt5_terminal
    ```

2.  **Configuration File:** Customize the `config.yaml` file to your desired settings for symbols, timeframe, features, models, risk management, etc.

### Running the Bot

To run the bot for live trading, execute the following command:

```bash
python main.py
```

### Training, Tuning, and Backtesting

*   **Training:** To train the initial models, you can use the `trainer.py` script.
*   **Tuning:** To tune the model hyperparameters, you can use the `tuner.py` script, which uses Optuna.
*   **Backtesting:** To backtest your strategies, you can use the `backtester.py` script.

## Development Conventions

*   **Code Structure:** The main source code is located in the `src` directory.
*   **Logging:** The project uses the `loguru` library for logging.
*   **Configuration:** The project uses `pyyaml` to manage the configuration and `python-dotenv` to manage environment variables.
*   **Hyperparameter Optimization:** The project uses `optuna` for hyperparameter optimization.
*   **Machine Learning:** The project uses `scikit-learn` for machine learning pipelines and utilities, and `pandas` and `numpy` for data manipulation.
*   **Technical Analysis:** The project uses the `ta` library for generating technical analysis features.

## Insights and Analysis

This section provides an analysis of the bot's design and capabilities based on its source code.

### How Good is This Bot?

The quality of this bot is highly dependent on the configuration, the quality of the data it's trained on, and the market conditions. However, its architecture suggests a high level of sophistication and a design that incorporates many best practices for algorithmic trading.

Compared to simpler, rule-based bots, this bot has the potential to be significantly more adaptive and profitable due to its use of machine learning. However, this also makes it more complex.

### Pros

*   **Sophisticated Modeling:** The use of an ensemble of gradient boosting models (LightGBM, XGBoost), Random Forest, and Logistic Regression is a powerful and robust approach. Ensembles typically outperform single models.
*   **Adaptive Risk Management:** The bot features a `RiskController` that uses Thompson Sampling, a machine learning technique in itself, to dynamically adjust risk parameters like stop-loss and take-profit levels. This allows the bot to adapt its risk profile to changing market conditions.
*   **High Configurability:** The `config.yaml` file allows for fine-tuning of nearly every aspect of the bot's operation, from feature engineering to model parameters and risk settings.
*   **Robust Development Lifecycle:** The inclusion of dedicated scripts for training, hyperparameter tuning (with Optuna), and backtesting demonstrates a mature development process, which is crucial for creating and validating effective trading strategies.
*   **Live Performance Monitoring:** The bot includes a `LivePerformanceMonitor` and can send Telegram notifications, allowing for real-time tracking of its performance and status.
*   **Data-Driven Feature Engineering:** The bot can generate a wide range of technical analysis features, and the selection of these features can be optimized during the tuning process.

### Cons

*   **Complexity:** The high level of sophistication and configurability also means the bot is complex. Users without a strong background in machine learning and Python may find it challenging to understand, configure, and maintain.
*   **No Graphical User Interface (GUI):** The bot is a command-line application, which may not be as user-friendly as a GUI-based application for some users.
*   **Data Dependency:** The performance of any machine learning system is heavily dependent on the quality and quantity of the data used for training. Poor data will lead to poor performance.
*   **Risk of Overfitting:** While the use of TimeSeriesSplit for cross-validation helps to mitigate overfitting, it remains a significant risk. An overfit model will perform well on historical data but poorly in live trading.
*   **Platform Dependency:** The bot is tightly integrated with MetaTrader 5, which limits its use to brokers that support the MT5 platform.

### Feedback and Recommendations

*   **Start with Backtesting:** Before deploying this bot with real money, it is **critical** to thoroughly backtest your strategies and configurations.
*   **Understand the Configuration:** Take the time to understand all the options in the `config.yaml` file. The default settings are a good starting point, but they may not be optimal for all market conditions or trading instruments.
*   **Monitor Performance:** Even after deployment, continuous monitoring of the bot's performance is essential. The live performance monitor and Telegram notifications are valuable tools for this.
*   **No Guaranteed Profits:** It is important to remember that no trading bot can guarantee profits. The financial markets are complex and unpredictable. This bot is a powerful tool, but it is not a "get rich quick" solution.