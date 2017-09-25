from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MnistPredictorBase(metaclass=ABCMeta):
    """
    Base class for mnist test data.
    """

    def __init__(self):
        self._sess = tf.Session()
        self._test_data = input_data.test

    def get_model_metadata_path(self):
        return self.get_model_checkpoint_path()+'.meta'

    @abstractmethod
    def get_model_checkpoint_path(self):
        raise NotImplementedError()

    def predict(self):
        saver = tf.train.import_meta_graph(self.get_model_metadata_path)
        saver.restore(self._sess, self.get_model_checkpoint_path())

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: self._test_data.images, y_: self._test_data.labels, keep_prob: 1.0}))
