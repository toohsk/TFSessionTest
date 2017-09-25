from predictor.mnist_adam_predictor import MnistAdamPredictor
from predictor.mnist_adadelta_predictor import MnistAdadeltaPredictor


if __name__ == '__main__':
    adam_predictor = MnistAdamPredictor()
    adam_predictor.predict()

    adadelta_predictor = MnistAdadeltaPredictor()
    adadelta_predictor.predict()