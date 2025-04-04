import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


class LSTM:
    def __init__(
            self,
            sequence_length=60,
            prediction_steps=10,
            train_split=0.95,
            features=None,
            target='Close',
            datapath='../datasets/BTC_ready.csv'
    ):
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.train_split = train_split
        self.features = features if features else ['Close']
        self.target = target

        self.data = self.load_data(datapath)
        self.model = self.build_model()

    def predict(self):
        scaled_data, scaled_target, _, scaler, target_scaler = self.preprocess_data(
            self.data,
            self.features,
            self.target
        )

        current_sequence = scaled_data[-self.sequence_length:].copy()
        input_sequence = current_sequence.reshape(1, self.sequence_length, len(self.features))

        pred_scaled = self.model.predict(input_sequence, verbose=0)

        pred_scaled_reshaped = pred_scaled.reshape(-1, 1)
        predictions = target_scaler.inverse_transform(pred_scaled_reshaped)
        return predictions

    def train(self, epochs=10, batch_size=32, verbose=False):
        scaled_data, scaled_target, training_data_len, scaler, target_scaler = self.preprocess_data(
            self.data,
            self.features,
            self.target
        )

        train_data = scaled_data[:training_data_len]
        train_target = scaled_target[:training_data_len]
        X_train, y_train = self.create_sequences(train_data, train_target)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=int(verbose))

        if verbose:
            test_data = scaled_data[training_data_len - self.sequence_length:]
            test_target = scaled_target[training_data_len - self.sequence_length:]
            X_test, y_test = self.create_sequences(test_data, test_target)

            predictions = self.model.predict(X_test[::self.prediction_steps])
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))

            selected_y = y_test[::self.prediction_steps]
            nsamples, nx, ny = selected_y.shape
            reshaped_y = selected_y.reshape(nsamples, nx * ny).reshape(-1, 1)
            actual = target_scaler.inverse_transform(reshaped_y)

            self.plot_predictions(actual, predictions)

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.features))))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(self.prediction_steps))

        model.compile(optimizer="adam",
                      loss="mae",
                      metrics=[keras.metrics.RootMeanSquaredError()])
        return model

    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(self.sequence_length, len(data) - self.prediction_steps + 1):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i:i + self.prediction_steps])
        return np.array(X), np.array(y)

    def preprocess_data(self, data, features, target='Close'):
        feature_data = data[features]
        target_data = data[[target]].values
        training_data_len = int(np.ceil(len(feature_data) * self.train_split))

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        target_scaler = StandardScaler()
        scaled_target = target_scaler.fit_transform(target_data)

        return scaled_data, scaled_target, training_data_len, scaler, target_scaler

    @staticmethod
    def plot_predictions(test, predictions):
        plt.figure(figsize=(12, 8))
        plt.plot(test, label='Test (Actual)', color='orange')
        plt.plot(predictions, label='Predictions', color='red')
        plt.title("Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()
        plt.show()

    @staticmethod
    def load_data(path):
        return pd.read_csv(path)


if __name__ == '__main__':
    features = [
        "Close", "Quote asset volume",
        "Number of trades", "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Score", "SMA_20", "RSI_14", "RSI_7", "MACD_14"
    ]

    model = LSTM(prediction_steps=20, features=features)
    model.train(20, verbose=True)
    plt.plot(model.predict())
    plt.title("Predictions")
    plt.show()
    #
    # scaled_data, scaled_target, training_data_len, _, target_scaler = model.preprocess_data(
    #     model.data, model.features, model.target
    # )
    #
    # test_data = scaled_data[training_data_len - model.sequence_length:]
    # test_target = scaled_target[training_data_len - model.sequence_length:]
    # X_test, y_test = model.create_sequences(test_data, test_target)
    #
    # predictions_scaled = model.model.predict(X_test, verbose=0)
    # predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    #
    # y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    #
    # mae = mean_absolute_error(y_test_actual, predictions)
    # mse = mean_squared_error(y_test_actual, predictions)
    # rmse = np.sqrt(mse)
    # mre = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
    #
    # print(f"MAE: {mae:.2f}")
    # print(f"MSE: {mse:.2f}")
    # print(f"RMSE: {rmse:.2f}")
    # print(f"Mean Relative Error: {mre:.2f}%")
