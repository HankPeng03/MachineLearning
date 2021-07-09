import functools

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

LABLE_COLUMN = "survived"
LABELS = [0, 1]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=LABLE_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

"""
examples, labels = next(iter(raw_train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)
"""

# 数据预处理

CATEGORIES = {
    "sex": ["male", "female"],
    "class": ["First", "Second", "Third"],
    "deck": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    "embark_town": ['Cherbourg', 'Southhampton', 'Queenstown'],
    "alone": ["y", "n"]
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature,
        vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))

def process_continuous_data(mean, data):
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, shape=[-1, 1])

MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

numerical_columns = []
for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(process_continuous_data, MEANS[feature]))
    numerical_columns.append(num_col)

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

train_data = raw_train_data.shuffle(500)
test_data = raw_test_data

# 训练
model.fit(train_data, epochs=20)

# 评估
test_loss, test_accuracy = model.evaluate(test_data)

test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))