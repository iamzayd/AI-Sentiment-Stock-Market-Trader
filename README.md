# AI Sentiment Stock Market Trader

### Developed by:
- **Ndubuisi Godcares Chibuokem** (21070126124)
- **Najeeb Saiyed** (21070126057)

## Project Overview
The **AI Sentiment Stock Market Trader** is an AI-powered trading bot that automates stock market trades by combining reinforcement learning with sentiment analysis. The bot utilizes real-time market data and sentiment analysis of news articles to make dynamic trading decisions aimed at maximizing returns while managing risk.

**GitHub Repository**: [AI Sentiment Stock Market Trader](https://github.com/iamzayd/AI-Sentiment-Stock-Market-Trader)

## Key Features
- **Reinforcement Learning (SARSA Algorithm)**: The SARSA algorithm dynamically decides whether to buy, sell, or hold stocks based on market conditions and sentiment analysis.
- **Sentiment Analysis**: Integrates the FinBERT model to analyze sentiment from news articles related to specific stocks.
- **Risk Management**: Implements position sizing and stop-loss strategies to protect against excessive losses.
- **Backtesting**: Provides the capability to backtest strategies using historical data from Yahoo Finance.
- **Automated Trading**: Connects to Alpaca's API to execute trades automatically based on the SARSA model's decisions and sentiment analysis.

## Architecture Overview
The architecture of the AI Sentiment Stock Market Trader consists of the following components:

- **Data Layer**: 
  - Fetches market data from Alpaca API.
  - Uses the FinBERT model to analyze sentiment from news articles related to specific stock symbols (e.g., SPY).
  
- **Algorithm Layer**: 
  - Implements the SARSA reinforcement learning algorithm to dynamically update the Q-table and guide trading decisions.
  - Analyzes news sentiment to refine trading strategies.
  
- **Execution Layer**: Executes trading decisions (buy, sell, or hold) using Alpaca’s API.

- **Backtesting Layer**: Backtests strategies using historical market data from Yahoo Finance to evaluate performance before live trading.

## Model Highlights
- **Increased Profitability**: Backtesting results demonstrate a cumulative return of 234% over four years, with an average of 1% profit per trade.
- **Adaptability**: The SARSA model adjusts to changing market conditions, utilizing both market data and sentiment analysis for more efficient trading.
- **Sentiment-Driven Decisions**: Analyzing sentiment from news articles (via FinBERT) helps predict stock price movements, improving decision-making accuracy.

## How It Works

1. **Sentiment Analysis**: 
   - The **news_classified.py** script fetches news articles from Alpaca’s API and analyzes them using FinBERT to classify their sentiment as positive, negative, or neutral.
   
2. **Reinforcement Learning (SARSA Algorithm)**:
   - The **SARSA1.py** script implements a SARSA-based trading strategy that evaluates market conditions and sentiment to make buy, sell, or hold decisions.
   - The Q-table is updated based on state-action-reward transitions as new market data and sentiment are gathered.

3. **Trading Execution**: 
   - Based on the SARSA model’s decisions, trades are executed via Alpaca’s API. The model also incorporates stop-loss and take-profit strategies to manage risks.

4. **Backtesting**: 
   - Backtesting is performed on historical data using Yahoo Finance to validate the strategy's effectiveness over time.

## Installation and Setup

### Clone the Repository:
```
git clone https://github.com/iamzayd/AI-Sentiment-Stock-Market-Trader.git
cd AI-Sentiment-Stock-Market-Trader
```

## API Keys

1. Create an account with [Alpaca](https://alpaca.markets/) to obtain API keys.

2. Add your Alpaca API keys to the respective scripts (`news_classified.py` and `SARSA1.py`).

   Make sure to update the API credentials in both scripts:
   - **news_classified.py**: Insert your API key and secret for fetching news.
   - **SARSA1.py**: Insert your API key and secret for executing trades.


## Run the Bot
### For Sentiment Analysis:

``` python news_classified.py```

### For Running the SARSA Trading Bot:

```python SARSA1.py```


## Usage

- **news_classified.py**: This script fetches news articles and classifies their sentiment using the FinBERT model. The results can be used to guide trading strategies.
- **SARSA1.py**: This script runs the SARSA trading bot, which automatically fetches market and sentiment data, makes trading decisions, and executes trades based on a predefined risk tolerance.

## Future Enhancements

- **Additional Sentiment Sources**: Extend sentiment analysis to include data from social media platforms (e.g., Twitter).
- **Strategy Diversification**: Support for trading a wider range of stocks and other financial instruments.
- **Improved Risk Management**: Implement advanced strategies such as volatility-based position sizing and dynamic stop-loss thresholds.

