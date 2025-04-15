import pandas as pd
import matplotlib.pyplot as plt

from TransformerModel.BTCPriceForecaster import BTCPriceForecaster
from model_training.dataset_autoupdate import update_btc_dataset

if __name__ == "__main__":
    forecaster = BTCPriceForecaster(data_path="../datasets/BTC_ready.csv")
    forecaster.load_and_prepare_data()
    forecaster.build_or_load_model()
    # forecaster.evaluate_model()
    future = forecaster.forecast(steps=24)
    timestamp_from = update_btc_dataset()
    # forecaster.load_and_prepare_data()
    # forecaster.train_model_on_new_data(timestamp_from)

    for item in future:
        print(item)

    timestamps = [pd.to_datetime(item["timestamp"], unit='s') for item in future]
    prices = [item["close"] for item in future]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, prices, marker='o', linestyle='-', color='b', label='Прогноз ціни BTC')
    plt.xlabel('Дата і час')
    plt.ylabel('Ціна закриття BTC')
    plt.title(f'Прогноз ціни BTC на {len(future)} годин вперед')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
