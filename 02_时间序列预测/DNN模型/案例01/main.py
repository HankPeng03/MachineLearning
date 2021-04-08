#coding:utf-8
import tensorflow as tf
import functools

LABLE_NAME = 'Flow_Add_0_Day'
COLUMN_NAMES = ['Flow_Add_-8_Day', 'Flow_Add_-7_Day',
       'Flow_Add_-6_Day', 'Flow_Add_-5_Day', 'Flow_Add_-4_Day',
       'Flow_Add_-3_Day', 'Flow_Add_-2_Day', 'Flow_Add_-1_Day',
       'Flow_Add_0_Day', 'Flow_Hour_0', 'Flow_Hour_9', 'Flow_Hour_14',
       'Flow_Hour_19', 'Flow_Add_0_Day_YoY', 'Year', 'Month', 'Day', 'Hour',
       'Dow', 'Woy', 'MonthC', 'DowC', 'HourC', 'Weather', 'Sun_Sub_2_Day',
       'Rain_Sub_2_Day', 'Sun_Sub_1_Day', 'Rain_Sub_1_Day', 'Sun', 'Rain',
       'Sun_Add_1_Day', 'Rain_Add_1_Day', 'Sun_Add_2_Day', 'Rain_Add_2_Day',
       'Is_Holiday_Sub_2_Day', 'Is_Holiday_Sub_1_Day', 'Is_Holiday',
       'Is_Holiday_Add_1_Day', 'Is_Holiday_Add_2_Day', 'LevelCode_event',
       'Price']

CATEGORICAL_COLUMN_NAMES = ['MonthC', 'DowC', 'HourC', 'Sun_Sub_2_Day',
       'Rain_Sub_2_Day', 'Sun_Sub_1_Day', 'Rain_Sub_1_Day', 'Sun', 'Rain',
       'Sun_Add_1_Day', 'Rain_Add_1_Day', 'Sun_Add_2_Day', 'Rain_Add_2_Day',
       'Is_Holiday_Sub_2_Day', 'Is_Holiday_Sub_1_Day', 'Is_Holiday',
       'Is_Holiday_Add_1_Day', 'Is_Holiday_Add_2_Day']

NUMERICAL_COLUMN_NAMES = ['Flow_Add_-8_Day', 'Flow_Add_-7_Day',
       'Flow_Add_-6_Day', 'Flow_Add_-5_Day', 'Flow_Add_-4_Day',
       'Flow_Add_-3_Day', 'Flow_Add_-2_Day', 'Flow_Add_-1_Day',
       'Flow_Hour_0', 'Flow_Hour_9', 'Flow_Hour_14',
       'Flow_Hour_19', 'Flow_Add_0_Day_YoY', 'Year', 'Month', 'Day', 'Hour',
       'Dow', 'Woy']


CATEGORIES = {'MonthC': ['g', 'e', 'a', 'd', 'b', 'c', 'f'],
                 'DowC': ['b', 'a', 'c'],
                 'HourC': ['e', 'a', 'd', 'b', 'c'],
                 'Sun_Sub_2_Day': [False, True],
                 'Rain_Sub_2_Day': [False, True],
                 'Sun_Sub_1_Day': [False, True],
                 'Rain_Sub_1_Day': [False, True],
                 'Sun': [False, True],
                 'Rain': [False, True],
                 'Sun_Add_1_Day': [False, True],
                 'Rain_Add_1_Day': [False, True],
                 'Sun_Add_2_Day': [False, True],
                 'Rain_Add_2_Day': [False, True],
                 'Is_Holiday_Sub_2_Day': [False, True],
                 'Is_Holiday_Sub_1_Day': [False, True],
                 'Is_Holiday': [False, True],
                 'Is_Holiday_Add_1_Day': [False, True],
                 'Is_Holiday_Add_2_Day': [False, True]}


NUMERICAL = {'Flow_Add_-8_Day': 64.40864179679969,
             'Flow_Add_-7_Day': 64.43543956043956,
             'Flow_Add_-6_Day': 64.46626180836707,
             'Flow_Add_-5_Day': 64.51653171390014,
             'Flow_Add_-4_Day': 64.54169076537498,
             'Flow_Add_-3_Day': 64.55345093502989,
             'Flow_Add_-2_Day': 64.57024773472142,
             'Flow_Add_-1_Day': 64.59726238673608,
             'Flow_Hour_0': 58.20487902619847,
             'Flow_Hour_9': 60.92933045701502,
             'Flow_Hour_14': 65.15023760204703,
             'Flow_Hour_19': 66.9460947503201,
             'Flow_Add_0_Day_YoY': 62.27750916972324,
             'Year': 2018.5288702525545,
             'Month': 6.221756313861577,
             'Day': 15.735010603431656,
             'Hour': 11.952188162714478,
             'Dow': 3.999566223250434,
             'Woy': 25.167245035666088}

def get_data(file_path):
    dataset = tf.data.experimental.make_csv_dataset(file_pattern=file_path,
                                                  batch_size=64,
                                                  select_columns=COLUMN_NAMES,
                                                  label_name=LABLE_NAME,
                                                  na_value='nan',
                                                  num_epochs=1,
                                                  ignore_errors=True)
    return dataset


def process_continuous_data(mean, data):
    data = 1/2 * tf.cast(data, tf.float32) / mean
    return tf.reshape(data, shape=[-1, 1])

if __name__ == '__main__':
    # 读取数据
    file_path = '/Users/PH/Workspace/GitHub/DeepLearning/SalesForecast/data/data.csv'
    data = get_data(file_path)
    # 处理类别型特征
    categorical_columns = list()
    for col_name, value in CATEGORIES.items():
        value = [str(i) for i in value]
        column = tf.feature_column.categorical_column_with_vocabulary_list(key=col_name,
                                                                           vocabulary_list=value,
                                                                           dtype=tf.string)
        categorical_columns.append(tf.feature_column.indicator_column(column))  # multi-hot 处理

    # 处理连续型特征
    numerical_columns = []
    for col_name, value in NUMERICAL.items():
        column = tf.feature_column.numeric_column(key=col_name,
                                                  dtype=tf.float32,
                                                  normalizer_fn=functools.partial(process_continuous_data, value))
        numerical_columns.append(column)

    # 构建预处理层
    preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numerical_columns)

    # 构建模型
    model = tf.keras.Sequential([
        preprocessing_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)])
    model.compile(loss=tf.keras.losses.mean_absolute_error,
                  optimizer='adam',
                  metrics=['accuracy'])
    # 训练与评估
    train_data = data.shuffle(500)
    model.fit(train_data, epochs=10)
    # 在验证数据集上的表现
    test_loss, test_accuracy = model.evaluate(train_data)