from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class LrCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
