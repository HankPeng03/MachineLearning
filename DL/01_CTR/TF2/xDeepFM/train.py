"""
Created on August 21, 2020

train xDeepFM model

@author: Ziyao Geng
"""

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_criteo_dataset
from model import xDeepFM

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = './data/criteo_sample/dac_sample.txt'
    read_part = True
    sample_num = 500000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]
    cin_size = [128, 128]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 2
    # ========================== Create dataset =======================
    feature_columns, train, test = create_criteo_dataset(file=file,
                                                         embed_dim=embed_dim,
                                                         read_part=read_part,
                                                         sample_num=sample_num,
                                                         test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = xDeepFM(feature_columns, hidden_units, cin_size)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/xdeepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Save==============================
    save_path = "./model/model"
    model.save_weights(save_path, save_format="tf")
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])