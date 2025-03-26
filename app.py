import requests
import pandas as pd
import time
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
from datetime import datetime
import pytz
import os

# Function to send messages to Telegram
def send_telegram_message(message):
    bot_token = "8122814466:AAGWq0gYOe5bxEG9dBuP1TgW5cRS_4f2A0"  # Telegram bot token
    chat_id = "246256619"  # Chat ID
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {"chat_id": chat_id, "text": message}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error sending message to Telegram: {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Error sending message to Telegram: {e}")
        return False

# Main loop for data collection and signal generation
prices_live = []
time_step = 12
analyzer = SentimentIntensityAnalyzer()
in_position = False  # Trade status
entry_price = 0  # Entry price
position_type = ""  # Trade type (buy or sell)
take_profit = 0  # Take profit
stop_loss = 0  # Stop loss
leverage = 5  # Leverage (can change to 10)
trades = []  # List of trades for daily report

# Function to detect Engulfing pattern
def detect_engulfing(df):
    if len(df) < 2:
        return "hold"
    prev_candle = df.iloc[-2]
    curr_candle = df.iloc[-1]
    # Bullish Engulfing
    if (prev_candle["Close"] < prev_candle["Close"] and
        curr_candle["Close"] > curr_candle["Close"] and
        curr_candle["Close"] > prev_candle["Close"] and
        curr_candle["Close"] < prev_candle["Close"]):
        return "buy"
    # Bearish Engulfing
    elif (prev_candle["Close"] > prev_candle["Close"] and
          curr_candle["Close"] < curr_candle["Close"] and
          curr_candle["Close"] < prev_candle["Close"] and
          curr_candle["Close"] > prev_candle["Close"]):
        return "sell"
    return "hold"

# Function for daily report
def daily_report():
    if not trades:
        report = "No trades were made in the past day."
        print(report)
        send_telegram_message(report)
        return report
    
    total_trades = len(trades)
    buy_trades = sum(1 for trade in trades if trade["type"] == "buy")
    sell_trades = sum(1 for trade in trades if trade["type"] == "sell")
    profitable_trades = sum(1 for trade in trades if trade["profit"] > 0)
    total_profit = sum(trade["profit"] for trade in trades)
    buy_profit = sum(trade["profit"] for trade in trades if trade["type"] == "buy")
    sell_profit = sum(trade["profit"] for trade in trades if trade["type"] == "sell")
    accuracy = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Find best and worst trades
    best_trade = max(trades, key=lambda x: x["profit"], default={"profit": 0, "timestamp": "N/A"})
    worst_trade = min(trades, key=lambda x: x["profit"], default={"profit": 0, "timestamp": "N/A"})
    
    report = f"Daily Trade Report ({datetime.now(pytz.timezone('Asia/Tehran')).strftime('%Y-%m-%d')}):\n"
    report += f"Total trades: {total_trades}\n"
    report += f"Buy trades: {buy_trades}\n"
    report += f"Sell trades: {sell_trades}\n"
    report += f"Profitable trades: {profitable_trades}\n"
    report += f"Signal accuracy: {accuracy:.2f}%\n"
    report += f"Total profit/loss: {total_profit:.2f}\n"
    report += f"Buy trades profit/loss: {buy_profit:.2f}\n"
    report += f"Sell trades profit/loss: {sell_profit:.2f}\n"
    report += f"Best trade: {best_trade['profit']:.2f} (Time: {best_trade['timestamp']})\n"
    report += f"Worst trade: {worst_trade['profit']:.2f} (Time: {worst_trade['timestamp']})\n"
    
    # Save report to file
    with open("daily_report.txt", "a") as f:
        f.write(report + "\n")
    
    # Send report to Telegram
    if send_telegram_message(report):
        print("Daily report successfully sent to Telegram.")
    else:
        print("Error sending daily report to Telegram.")
    
    print(report)
    return report

last_report_time = None

# Liara runs the script as a background worker
def main():
    global prices_live, in_position, entry_price, position_type, take_profit, stop_loss, trades, last_report_time
    while True:
        # Fetch real-time price and volume
        url = "https://api.coinlore.net/api/ticker/?id=90"
        response = requests.get(url)
        data = response.json()
        if data:
            price = float(data[0]["price_usd"])
            volume = float(data[0]["volume24"])
            timestamp = pd.Timestamp.now()
            prices_live.append({"timestamp": timestamp, "Close": price, "Volume": volume})
            print(f"New data: {timestamp}, Price: {price}, Volume: {volume}")

        # Calculate indicators
        if len(prices_live) >= 50:  # Minimum 50 data points for MA50
            df_live = pd.DataFrame(prices_live)
            df_live["RSI"] = ta.momentum.RSIIndicator(df_live["Close"]).rsi()
            df_live["MACD"] = ta.trend.MACD(df_live["Close"]).macd()
            df_live["BB_upper"] = ta.volatility.BollingerBands(df_live["Close"]).bollinger_hband()
            df_live["BB_lower"] = ta.volatility.BollingerBands(df_live["Close"]).bollinger_lband()
            df_live["Stochastic"] = ta.momentum.StochasticOscillator(df_live["Close"], df_live["Close"], df_live["Close"]).stoch()
            df_live["MA50"] = df_live["Close"].rolling(window=50).mean()
            df_live["MA200"] = df_live["Close"].rolling(window=200).mean()
            df_live["ATR"] = ta.volatility.AverageTrueRange(df_live["Close"], df_live["Close"], df_live["Close"]).average_true_range()

            # Since we don't have the model, we'll skip price prediction and use indicators
            current_price = df_live["Close"].iloc[-1]

            # Price signal (based on MA crossover instead of prediction)
            ma50 = df_live["MA50"].iloc[-1]
            ma200 = df_live["MA200"].iloc[-1] if len(df_live) >= 200 else ma50
            price_signal = "buy" if ma50 > ma200 else "sell"

            # RSI signal
            rsi = df_live["RSI"].iloc[-1]
            rsi_signal = "buy" if rsi < 30 else "sell" if rsi > 70 else "hold"

            # Stochastic signal
            stochastic = df_live["Stochastic"].iloc[-1]
            stochastic_signal = "buy" if stochastic < 20 else "sell" if stochastic > 80 else "hold"

            # MA signal
            ma_signal = "buy" if ma50 > ma200 else "sell"

            # Volume signal
            volume = df_live["Volume"].iloc[-1]
            avg_volume = df_live["Volume"].mean()
            volume_signal = "buy" if volume > avg_volume else "sell"

            # Price action signal (Engulfing)
            price_action_signal = detect_engulfing(df_live)

            # Fetch and analyze news
            url = "https://cryptopanic.com/api/v1/posts/?auth_token=b2efe219042d1df5203166c055ba4dcb1cb95b88¤cies=BTC"
            response = requests.get(url)
            news_data = response.json()
            news_sentiment = []
            if "results" in news_data:
                for post in news_data["results"]:
                    sentiment = analyzer.polarity_scores(post["title"])["compound"]
                    news_sentiment.append(sentiment)
                avg_sentiment = sum(news_sentiment) / len(news_sentiment) if news_sentiment else 0
                print(f"Average news sentiment: {avg_sentiment}")
            else:
                avg_sentiment = 0
                print("Error fetching news: ", news_data)

            # News signal
            news_signal = "buy" if avg_sentiment > 0.5 else "sell" if avg_sentiment < -0.5 else "hold"

            # Combine signals with weights
            signals = {
                "price": (price_signal, 2),  # Weight 2
                "rsi": (rsi_signal, 1),
                "stochastic": (stochastic_signal, 1),
                "ma": (ma_signal, 1),
                "volume": (volume_signal, 1),
                "price_action": (price_action_signal, 1),
                "news": (news_signal, 1)
            }
            buy_score = sum(weight for signal, weight in signals.values() if signal == "buy")
            sell_score = sum(weight for signal, weight in signals.values() if signal == "sell")
            final_signal = "Enter (buy)" if buy_score >= 4 else "Exit (sell)" if sell_score >= 4 else "Hold"
            signal_message = (f"New data: {df_live['timestamp'].iloc[-1]}, Price: {current_price}, Volume: {volume}\n"
                             f"Signals: {dict((k, v[0]) for k, v in signals.items())}\n"
                             f"Buy score: {buy_score}, Sell score: {sell_score}\n"
                             f"Final signal: {final_signal}")
            print(signal_message)
            send_telegram_message(signal_message)

            # Trade management
            atr = df_live["ATR"].iloc[-1]  # ATR for stop loss and take profit
            if not in_position:
                # Enter trade
                if final_signal == "Enter (buy)":
                    in_position = True
                    position_type = "buy"
                    entry_price = current_price
                    stop_loss = entry_price - 2 * atr  # Stop loss: 2x ATR below entry
                    take_profit = entry_price + 4 * atr  # Take profit: 4x ATR above entry (1:2 ratio)
                    entry_message = (f"Enter buy trade: Entry price: {entry_price}, "
                                    f"Stop loss: {stop_loss}, Take profit: {take_profit}")
                    print(entry_message)
                    send_telegram_message(entry_message)
                elif final_signal == "Exit (sell)":
                    in_position = True
                    position_type = "sell"
                    entry_price = current_price
                    stop_loss = entry_price + 2 * atr  # Stop loss: 2x ATR above entry
                    take_profit = entry_price - 4 * atr  # Take profit: 4x ATR below entry (1:2 ratio)
                    entry_message = (f"Enter sell trade: Entry price: {entry_price}, "
                                    f"Stop loss: {stop_loss}, Take profit: {take_profit}")
                    print(entry_message)
                    send_telegram_message(entry_message)
            else:
                # Manage open trade
                if position_type == "buy":
                    profit = (current_price - entry_price) * leverage  # Profit with leverage
                    if current_price <= stop_loss:
                        close_message = f"Trade closed (stop loss): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "buy", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    elif current_price >= take_profit:
                        close_message = f"Trade closed (take profit): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "buy", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    elif final_signal == "Exit (sell)":
                        close_message = f"Trade closed (opposite signal): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "buy", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    else:
                        # Check for take profit update
                        new_take_profit = current_price + 2 * atr  # New take profit
                        if new_take_profit > take_profit:
                            take_profit = new_take_profit
                            update_message = f"Take profit updated: {take_profit}"
                            print(update_message)
                            send_telegram_message(update_message)
                        status_message = (f"Trade status: Current profit/loss: {profit}, "
                                         f"Take profit: {take_profit}, Stop loss: {stop_loss}")
                        print(status_message)
                        send_telegram_message(status_message)
                elif position_type == "sell":
                    profit = (entry_price - current_price) * leverage  # Profit with leverage
                    if current_price >= stop_loss:
                        close_message = f"Trade closed (stop loss): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "sell", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    elif current_price <= take_profit:
                        close_message = f"Trade closed (take profit): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "sell", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    elif final_signal == "Enter (buy)":
                        close_message = f"Trade closed (opposite signal): Profit/loss: {profit}"
                        print(close_message)
                        send_telegram_message(close_message)
                        trades.append({"type": "sell", "profit": profit, "timestamp": df_live["timestamp"].iloc[-1]})
                        in_position = False
                    else:
                        # Check for take profit update
                        new_take_profit = current_price - 2 * atr  # New take profit
                        if new_take_profit < take_profit:
                            take_profit = new_take_profit
                            update_message = f"Take profit updated: {take_profit}"
                            print(update_message)
                            send_telegram_message(update_message)
                        status_message = (f"Trade status: Current profit/loss: {profit}, "
                                         f"Take profit: {take_profit}, Stop loss: {stop_loss}")
                        print(status_message)
                        send_telegram_message(status_message)

            # Daily report (every day at midnight Tehran time)
            tehran_tz = pytz.timezone('Asia/Tehran')
            current_time = datetime.now(tehran_tz)
            if (last_report_time is None or
                    (current_time.hour == 0 and current_time.minute < 5 and
                     last_report_time.day != current_time.day)):
                report = daily_report()
                last_report_time = current_time
                trades = []  # Reset trades list for the new day

        # Save data
        df_live = pd.DataFrame(prices_live)
        df_live.to_csv("btc_live.csv", index=False)

        # Wait 5 minutes
        time.sleep(300)

if __name__ == "__main__":
    main()
