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
            train_split=0.95,
            features=None,
            target='Close',
            datapath='../datasets/BTC_ready.csv'
    ):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.features = features if features else ['Close']
        self.target = target

        self.data = self.load_data(datapath)
        self.model = self.build_model()

    def predict(self, num_rows):
        scaled_data, scaled_target, _, scaler, target_scaler = self.preprocess_data(
            self.data,
            self.features,
            self.target
        )

        current_sequence = scaled_data[-self.sequence_length:].copy()

        predictions_scaled = []

        for _ in range(num_rows):
            input_sequence = current_sequence.reshape(1, self.sequence_length, len(self.features))
            pred_scaled = self.model.predict(input_sequence, verbose=0)
            predictions_scaled.append(pred_scaled[0, 0])

            new_row = np.array([[pred_scaled[0, 0]]])
            current_sequence = np.append(current_sequence, new_row, axis=0)
            current_sequence = current_sequence[1:]

        predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions_scaled)
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

            predictions = self.model.predict(X_test)
            predictions = target_scaler.inverse_transform(predictions)

            test = self.data[training_data_len:].copy()
            test['Predictions'] = predictions
            self.plot_predictions(test, predictions)

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.features))))
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        model.compile(optimizer="adam",
                      loss="mae",
                      metrics=[keras.metrics.RootMeanSquaredError()])
        return model

    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        X, y = np.array(X), np.array(y)
        return X, y

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
    def load_data(path):
        return pd.read_csv(path)

    @staticmethod
    def plot_predictions(test, predictions, date_column='Open time', target_column='Close'):
        plt.figure(figsize=(12, 8))
        plt.plot(test[date_column], test[target_column], label='Test (Actual)', color='orange')
        plt.plot(test[date_column], predictions, label='Predictions', color='red')
        plt.title("Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    model = LSTM()
    model.train(20, verbose=True)
    # predictions = model.predict(10)
    # plt.figure(figsize=(12, 8))
    # plt.plot(predictions)
    # plt.show()
