from instanceseg.train import trainer


class Evaluator(trainer.Trainer):
    def __init__(self, cuda, model, **kwargs):
        super(self, Evaluator).__init__(cuda, model, **kwargs)
