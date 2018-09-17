import numpy as np
import pandas as pd

data_path = 'Shanghai Shenzhen CSI 300 Historical Data.csv'
class data_tool(object):

    def __init__(self, data_path, split_ratio):
        # load time series
        self.data = np.array([float(i.replace(',', '')) for i in pd.read_csv(data_path)['Price'][::-1]])

        #
        window_size = 21

        #
        raw = []
        for i in range(self.data.shape[0] - window_size):
            raw.append(self.data[i: i + window_size])

        # normalized data by dividing first element
        self.norm_raw = self.normalize(raw)
        self.data_raw = np.array(raw)
        self.data = np.array(self.norm_raw)

        # split train/test
        self.split_point = int(split_ratio * self.data.shape[0])
        self.train = self.data[:self.split_point]
        self.test = self.data[self.split_point:]

        # separate x and y
        self.train_x = self.train[:, :-1]
        self.train_y = self.train[:, -1].reshape([-1, 1])

        self.test_x = self.test[:, :-1]
        self.test_y = self.test[:, -1].reshape([-1, 1])

        # raw data
        self.test_raw_x = self.data_raw[self.split_point:, :-1]
        self.test_raw_y = self.data_raw[self.split_point:, -1]

    def generate_batches(self, data, num_epoch, batch_size, shuffle=True):
        data = np.array(data)
        data_size = len(data)

        num_batches_per_epoch = data_size // batch_size + 1

        for epoch in range(num_epoch):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def normalize(self, data):
        normalized_data = []
        for window in data:
            normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalized_data.append(normalized_window)
        return normalized_data

