import numpy as np
import tensorflow as tf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000, ))
x_val = np.random.random((200, 32))
y_val = np.random.randint(10, size=(200, ))
x_test = np.random.random((200, 32))
y_test = np.random.randint(10, size=(200, ))

def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model


model = get_compiled_model()
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))



tf.Variable

tf.keras.optimizers.Adam._decayed_lr()
tf.keras.callbacks.EarlyStopping


class LearningRateMonitor(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}, model:{}".format(
            epoch, keys, tf.keras.backend.get_value(self.model.optimizer.learning_rate)))

    def on_train_batch_begin(self, batch, logs=None):
        print("...Training: start of batch {}; got learning rate: {}".format(
            batch, tf.keras.backend.get_value(self.model.optimizer.learning_rate)))












