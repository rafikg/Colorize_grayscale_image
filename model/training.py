import tensorflow as tf
import datetime
import logging
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard, EarlyStopping)

from data import ColorizeDataset
from model import ColorizeGrayScale
from config import reg

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Define distribution strategy
strategy = tf.distribute.MirroredStrategy()

# logger info device

logger.info(
    'Training will run on {} GPUs:'.format(strategy.num_replicas_in_sync))

# Select the batch size per replica
BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Select epochs
EPOCHS = 10

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=1e-3)

logger.info('Start training')

# Define the dataset object
dataset_train_obj = ColorizeDataset(
    path="../dataset/train_data",
    img_ext="*.jpg",
    batch_size=BATCH_SIZE,
    debug_mode=False,
    n_workers=8)

dataset_valid_obj = ColorizeDataset(
    path="../dataset/valid_data",
    img_ext="*.jpg",
    batch_size=BATCH_SIZE,
    n_workers=8,
    debug_mode=False,
    is_validation=True,
    is_training=False)

len_train = len(dataset_train_obj)
len_valid = len(dataset_valid_obj)

logger.info("Training on dataset with size {}".format(len_train))
logger.info("Validate on dataset with size {}".format(len_valid))

train_dataset = dataset_train_obj.tf_data
valid_dataset = dataset_valid_obj.tf_data

# Define the model inside a strategy.scope
with strategy.scope():
    # create the model
    model = ColorizeGrayScale(l2_reg=reg)
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

# Define Callbacks list
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '../logs/' + current_time + '/train'

model_tag = '../model_weights/model.{epoch:02d}'

tensorboard = TensorBoard(log_dir=train_log_dir,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True,
                          update_freq='epoch')

model_checkpoint = ModelCheckpoint(filepath=model_tag, monitor='loss',
                                   verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=10)
early_stop = EarlyStopping('val_loss', mode='min', patience=10)

callbacks = [tensorboard, model_checkpoint, reduce_lr, early_stop]

# Start training the model
model.fit(x=train_dataset,
          steps_per_epoch=None,
          validation_data=valid_dataset,
          validation_steps=None,
          epochs=EPOCHS,
          callbacks=callbacks)
model.summary()
