from .predictor_base import MnistPredictorBase


class MnistAdadeltaPredictor(MnistPredictorBase):

    # override
    def get_model_checkpoint_path(self):
        return "tmp/"

    