# encoding=utf8
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def df_to_dataset(df, shuffle, batch_size):
    labels = df.loc[:, "target"]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=df.shape[0])
    ds = ds.batch(batch_size)
    return ds

if __name__=="__main__":
    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    df = pd.read_csv(URL)
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    # 用tf.data 创建输入pipline
    batch_size = 5
    train_ds = df_to_dataset(train, shuffle=True, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # 特征处理
    feature_columns = list()

    # 数值列
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(feature_column.numeric_column(col))
    # 分桶列
    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    feature_columns.append(age_buckets)

    # 分类列
    thal = feature_column.categorical_column_with_vocabulary_list(
                          "thal", ["fixed", "normal", "reversible"])
    thal_one_hot = feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    # 嵌入列
    thal_embedding = feature_column.embedding_column(thal, dimension=8)
    feature_columns.append(thal_embedding)

    # 组合列
    crossed_feature = feature_column.crossed_column(keys=[age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    # 建立特征层
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(units=128, activation="relu"),
        layers.Dense(units=128, activation="relu"),
        layers.Dense(units=1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=True)
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)