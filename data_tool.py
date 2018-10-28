import numpy as np
import pandas as pd

data_path = 'Shanghai Shenzhen CSI 300 Historical Data.csv'
class data_tool(object):

    def __init__(self, data_path, split_ratio, window_size=5):
        # load time series
        self.data = np.array([float(i.replace(',', '')) for i in pd.read_csv(data_path)['Price'][::-1]])

        # construct sequencies
        raw = []
        for i in range(self.data.shape[0] - window_size):
            raw.append(self.data[i: i + window_size])
        self.data_raw = np.array(raw)

        # normalized data by dividing first element
        self.norm_raw = self.normalize(self.data_raw)
        self.data = np.array(self.norm_raw)

        # split train/test
        self.split_point = int(split_ratio * self.data.shape[0])
        self.train = self.data[:self.split_point]
        self.test = self.data[self.split_point:]

        # separate x and y
        self.train_x = self.train[:, :-1]
        self.test_x = self.test[:, :-1]

        self.train_y = self.train[:, -1].reshape([-1, 1])
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
        return (data / data[:, 0].reshape((-1, 1))) - 1


if __name__ == '__main__':
    test = data_tool(data_path, split_ratio=0.7)