# encoding="utf8"
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

if __name__=="__main__":
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    df = pd.read_csv(csv_file)
    df.loc[:, "target"] = np.where(df.loc[:, "AdoptionSpeed"]==4, 0, 1)
    df = df.drop(columns=['AdoptionSpeed', 'Description'])

    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')


    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(df, shuffle=True, batch_size=32):
        df = df.copy()
        labels = df.pop('target')
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)


    def get_normalization_layer(name, dataset):
        normalizer = preprocessing.Normalization(axis=None)
        feature_ds = dataset.map(lambda x, y: x[name])
        normalizer.adapt(feature_ds)
        return normalizer

    layer = get_normalization_layer("PhotoAmt", train_ds)

    def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
        """
         returns a layer which maps values from a vocabulary to integer indices and one-hot encodes the features.
        :param name:
        :param dataset:
        :param dtype:
        :param max_tokens:
        :return:
        """
        if dtype == "string":
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(vocabulary_size=max_tokens)

        feature_ds = dataset.map(lambda x, y: x[name])
        index.adapt(feature_ds)
        encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
        return lambda feature: encoder(index(feature))

    all_inputs = []
    encoded_features = []

    for col in ["PhotoAmt", "Fee"]:
        numeric_col = tf.keras.Input(shape=(1,), name=col, dtype="float")
        normalization_layer = get_normalization_layer(col, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    age_col = tf.keras.Input(shape=(1,), name="Age", dtype="int64")
    encoding_layer = get_category_encoding_layer(name="Age", dataset=train_ds, dtype="int64", max_tokens=5)
    encoded_age_col = encoding_layer(age_col)
    all_inputs.append(age_col)
    encoded_features.append(encoded_age_col)

    categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                        'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
    for col in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=col, dtype="string")
        encoding_layer = get_category_encoding_layer(name=col, dataset=train_ds, dtype="string", max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(units=32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

