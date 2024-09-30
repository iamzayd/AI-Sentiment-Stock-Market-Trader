import numpy as np
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment

#ALPACA Credentials
API_KEY = ####
API_SECRET = ####
BASE_URL = #####

# Dictionary for Alpaca credentials
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

#SARSA Strategy Function
class SARSATrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.9, sleeptime: str = "24H"):
        self.symbol = symbol
        self.sleeptime = sleeptime
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = ['buy', 'sell', 'hold']
        self.states = ['positive', 'neutral', 'negative']
        self.Q = {}  # Q-table
        
    # Method for position sizing - determines how much of the stock to buy/sell
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol) # Get the last price of the stock
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    # Method to calculate the date range (last 3 days for news analysis)
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3) #Calculate the date 3 days ago
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    # Method to get sentiment analysis from news articles using the FinBERT model
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment
    
    # Method to choose the next action (buy/sell/hold) based on the Q-table or randomly (exploration)
    def choose_action(self, state):
        if state not in self.Q:
            return np.random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    # Update the Q-table based on the SARSA algorithm
    def update_Q(self, state, action, reward, next_state, next_action):
        if state not in self.Q:
            self.Q[state] = {'buy': 0, 'sell': 0, 'hold': 0}
        if next_state not in self.Q:
            self.Q[next_state] = {'buy': 0, 'sell': 0, 'hold': 0}
        self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

    # Calculate reward based on the price difference after executing an action
    def calculate_reward(self, action, last_price, next_last_price):
        if action == 'buy':
            reward = next_last_price - last_price
        elif action == 'sell':
            reward = last_price - next_last_price
        else:
            reward = 0 # No reward for holding the stock
        return reward
    
    # Trading iteration method
    def on_trading_iteration(self):
        
        # Get the available cash, last price of the stock, and how much can be traded
        cash, last_price, quantity = self.position_sizing()
        
        # Get the sentiment for the stock from recent news
        probability, sentiment = self.get_sentiment()
        state = (cash, last_price, sentiment) #Defines the current state
        
        # If the state is not in the Q-table, initialize it
        if state not in self.Q:
            self.Q[state] = {'buy': 0, 'sell': 0, 'hold': 0}

         # Here the model chooses action based on exploration (epsilon) or exploitation (Q-values)
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = max(self.Q[state], key=self.Q[state].get)  # Chooses best action based on Q-values

        
        # Execute the chosen action (buy/sell/hold)
        if quantity > 0:
            if action == 'buy':
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20, # Set a 20% profit target
                    stop_loss_price=last_price * 0.95   #Setting a 5% stop loss
                )
                self.submit_order(order)
                self.last_trade = "buy"
                
            elif action == 'sell':
                if self.last_trade == "buy":
                    self.sell_all()
                    
                    # Bracket order with take-profit and stop-loss limits
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"
            elif action == "hold":
                pass # No action taken for 'hold'
        else:
            action = "hold" # Default to hold is no quatity is available

         # Update the next state and Q-values
        next_cash, next_last_price, _ = self.position_sizing()
        reward = self.calculate_reward(action, last_price, next_last_price)

        next_probability, next_sentiment = self.get_sentiment()
        next_state = (next_cash, next_last_price, next_sentiment)

         # Choose the next action based on the next state
        if np.random.rand() < self.epsilon:
            next_action = np.random.choice(self.actions)
        else:
            next_action = max(self.Q[next_state], key=self.Q[next_state].get)
            
        # Update the Q-values using the SARSA update rule
        self.update_Q(state, action, reward, next_state, next_action)

#Backtesting
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 7, 31)
broker = Alpaca(ALPACA_CREDS)
strategy = SARSATrader(name='sarsa_strat', broker=broker, parameters={"symbol": "SPY", "cash_at_risk": 0.5, "sleeptime": "24H"})  
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)
