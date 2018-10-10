from data_tool import data_tool
from RNNModel import RNNModel
import tensorflow as tf
import datetime, os
import matplotlib.pyplot as plt
import pandas as pd

data_path = 'Shanghai Shenzhen CSI 300 Historical Data.csv'
class train(data_tool, RNNModel):

    def __init__(self):
        data_tool.__init__(self, data_path=data_path, split_ratio=0.8)
        self.batch_size = 64
        self.epoch_size = 20

        with tf.Graph().as_default():
            RNNModel.__init__(self, sequence_length=20, RNN_size=100)
            sess = tf.Session()
            with sess.as_default():

                global_step = tf.Variable(0, name='global_step', trainable=False)

                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step)

                # generate folder for summaries
                summary_dir = str(int(datetime.datetime.now().timestamp()))

                # Summary for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)

                # Train Summaries
                train_summary_op = loss_summary
                train_summary_dir = os.path.join("runs", summary_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Test Summaries
                test_summary_op = loss_summary
                test_summary_dir = os.path.join("runs", summary_dir, 'summaries', 'test')
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

                # define operations
                def train_(batch_x, batch_y):
                    feed_dict = {self.input_x: batch_x,
                                 self.output_y: batch_y,
                                 self.keep_prob: 0.5,
                                 }

                    loss, _, step, summary = sess.run(
                        [self.loss, train_op, global_step, train_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, MAE {:g}".format(time_str, step, loss))
                    train_summary_writer.add_summary(summary, step)


                def test_(return_output=False):
                    feed_dict = {self.input_x: self.test_x,
                                 self.output_y: self.test_y,
                                 self.keep_prob: 1.0,
                                 }
                    test_output, loss, step, summary = sess.run(
                        [self.output, self.loss, global_step, test_summary_op],
                        feed_dict=feed_dict)

                    time_str = datetime.datetime.now().isoformat()
                    print("Test: {}: step {}, MAE {:g}".format(time_str, step, loss))
                    test_summary_writer.add_summary(summary, step)

                    if return_output:
                        return test_output
                    else:
                        pass


                # initialize variable
                sess.run(tf.global_variables_initializer())

                # generate batches
                batches_all = self.generate_batches(data=list(zip(self.train_x, self.train_y)),
                                                    num_epoch=self.epoch_size,
                                                    batch_size=self.batch_size, shuffle=True)
                total_amount = (len(self.train_x) // self.batch_size + 1) * self.epoch_size
                print(total_amount)
                for i, batch in enumerate(batches_all):
                    batch_x, batch_y = zip(*batch)
                    train_(batch_x, batch_y)

                    if i % 10 == 0:
                        print('\nEvaluation:\n')
                        test_()

                self.test_output = test_(True)

    def Evaluation(self, plot=False):
        # get predicted data
        print(self.test_output.shape)
        predicted_value = [(self.test_output[i] + 1) * self.test_raw_x[i][0] for i in range(len(self.test_raw_x))]

        tmp = pd.DataFrame(list(zip(self.test_raw_y, predicted_value)), columns=['Real', 'Predicted'])
        tmp.to_csv('result.csv', index=False)
        # plot
        if plot:
        	plt.subplot(111)
        	plt.plot(predicted_value, label="Predicted")
        	plt.plot(self.test_raw_y.tolist(), label='Actual')
        	plt.legend()
        	plt.show()



if __name__ == '__main__':
    test = train()
    test.Evaluation()