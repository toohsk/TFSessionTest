from .predictor_base import MnistPredictorBase


class MnistAdamPredictor(MnistPredictorBase):

    # override
    def get_model_checkpoint_path(self):
        return "/tmp/model/adam/model.ckpt"

    