import streamlit as st
from lppls import lppls
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
import ccxt
import requests
import matplotlib.pyplot as plt
import time
from multiprocessing import freeze_support

# Initialize the Binance exchange API
exchange = ccxt.kucoin()

def send_to_telegram(message):
    apiToken = '5888707741:AAHkGp2FW3-Lrs7GxmXpbtfv7LCltgcIQb4'
    chatID = '-1001884976731'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)

def ai_modelling():
    # Define the trading pair and timeframe
    symbol = 'BTC/USDT'
    timeframe = '1h'  # Daily candles

    # Fetch historical price data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    # Convert the Pandas datetime objects to a date format (removing hours, minutes, and seconds)
    data['timestamp'] = data['timestamp'].dt.date

    # Set 'date_only' as the index and remove the numeric index
    data.set_index('timestamp', drop=False)
    data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False)

    # Convert time to ordinal
    data['timestamp'] = [pd.Timestamp.toordinal(dt.strptime(str(t1), '%Y-%m-%d')) for t1 in data['timestamp']]

    # Create list of observation data
    data['logclose'] = np.log(np.flip(data['close'].values))
    price = data['logclose']

    # Create observations array (expected format for LPPLS observations)
    observations = np.array([data['timestamp'], price])

    # Set the max number for searches to perform before giving up
    MAX_SEARCHES = 25

    # Instantiate a new LPPLS model with the BTC/USDT dataset
    lppls_model = lppls.LPPLS(observations=observations)

    # Fit the model to the data and get back the params
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)

    """
    Save Fit
    Save and show your fitted results
    """
    time_ord = [pd.Timestamp.fromordinal(d) for d in lppls_model.observations[0, :].astype('int32')]
    t_obs = lppls_model.observations[0, :]
    lppls_fit = [lppls_model.lppls(t, tc, m, w, a, b, c1, c2) for t in t_obs]  # ------ FITTED MOVING AVERAGE
    price = lppls_model.observations[1, :]
    true_price = [val for val in data['close']]
    """
    Save Confidence Indicator
    Run computations for lppl
    """
    # compute the confidence indicator
    res = lppls_model.mp_compute_nested_fits(
        workers=8,
        window_size=120,
        smallest_window_size=30,
        outer_increment=1,
        inner_increment=5,
        max_searches=25,
        # filter_conditions_config={} # not implemented in 0.6.x
    )
    res_df = lppls_model.compute_indicators(res)
    positive_confidence = [val for val in res_df['pos_conf']]
    negative_confidence = [val for val in res_df['neg_conf']]

    return true_price, lppls_fit, positive_confidence, negative_confidence

def is_new_hour():
    current_hour = dt.now().hour
    time.sleep(1)  # Sleep for a short time to avoid multiple reloads within the same hour
    return current_hour != dt.now().hour

if __name__ == "__main__":
    freeze_support()

    while True:
        if is_new_hour():
            st.experimental_rerun()
        
        price, lppls_fit, positive_confidence, negative_confidence = ai_modelling()

        # Create a figure with four subplots
        fig, axs = plt.subplots(4, 1, figsize=(15, 12))

        # Plot on the first subplot
        axs[0].plot(price, color='blue')
        axs[0].set_title('LOG PRICE')

        # Plot on the second subplot
        axs[1].plot(lppls_fit, color='purple')
        axs[1].set_title('FITTED MA')

        # Plot on the third subplot
        axs[2].plot(positive_confidence, color='green')
        axs[2].set_title('POSITIVE CONFIDENCE')

        # Plot on the fourth subplot
        axs[3].plot(negative_confidence, color='red')
        axs[3].set_title('NEGATIVE CONFIDENCE')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the Matplotlib chart using Streamlit
        st.pyplot(fig)
