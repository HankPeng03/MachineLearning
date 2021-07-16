import tensorflow as tf
import os
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


if __name__=="__main__":
    df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')

    cleaned_df = df.copy()

    # You don't want the `Time` column.
    cleaned_df.pop('Time')

    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount') + eps)

    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_labels = np.array(train_df.pop("Class"))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop("Class"))
    test_labels = np.array(test_df.pop("Class"))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR')
    ]

    def make_model(metrics=METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = keras.Sequential([
          keras.layers.Dense(
              16, activation='relu',
              input_shape=(train_features.shape[-1],)),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(1, activation='sigmoid',
                             bias_initializer=output_bias),
        ])

        model.compile(
          optimizer=keras.optimizers.Adam(lr=1e-3),
          loss=keras.losses.BinaryCrossentropy(),
          metrics=metrics)

        return model


    EPOCHS = 100
    BATCH_SIZE = 2048

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model = make_model()

    model.predict(train_features[:10])

    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))

    neg, pos = np.bincount(df['Class'])
    initial_bias = np.log([pos / neg])
    initial_bias

    model = make_model(output_bias=initial_bias)
    model.predict(train_features[:10])

    results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))

    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)


    model = make_model()
    model.load_weights(initial_weights)
    model.layers[-1].bias.assign([0.0])
    zero_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(val_features, val_labels),
        verbose=0)

    model = make_model()
    model.load_weights(initial_weights)
    careful_bias_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(val_features, val_labels),
        verbose=0)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_loss(history, label, n):
        # Use a log scale on y-axis to show the wide range of values.
        plt.semilogy(history.epoch, history.history['loss'],
                     color=colors[n], label='Train ' + label)
        plt.semilogy(history.epoch, history.history['val_loss'],
                     color=colors[n], label='Val ' + label,
                     linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')


    plot_loss(zero_bias_history, "Zero Bias", 0)
    plot_loss(careful_bias_history, "Careful Bias", 1)

    model = make_model()
    model.load_weights(initial_weights)
    baseline_history = model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        validation_data=(val_features, val_labels))


    def plot_metrics(history):
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_", " ").capitalize()
            plt.subplot(2, 2, n + 1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8, 1])
            else:
                plt.ylim([0, 1])

            plt.legend()


    plot_metrics(baseline_history)


