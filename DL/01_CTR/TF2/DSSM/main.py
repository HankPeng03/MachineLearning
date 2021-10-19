#encoding=utf8
from utils import *
from model import DSSM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import os
import datetime

# 训练集
filenames = tf.data.Dataset.list_files([os.path.join(base_dir, 'data', 'recall_user_item_act.csv')])
dataset = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 8
dataset = dataset.map(parse_function, num_parallel_calls=60)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size=batch_size * 2)  # 在缓冲区中随机打乱数据

pad_shapes = {}
pad_values = {}

for feat_col in feature_columns:
    if isinstance(feat_col, SparseFeat):
        if feat_col.dtype == 'string':
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = '9999'
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0.0

# tf.TensorShape([]) 表示长度为单个数字
# pad_shapes = (pad_shapes, (tf.TensorShape([])))
# pad_values = (pad_values, (tf.constant('', dtype=tf.float32)))

dataset = dataset.padded_batch(batch_size=batch_size)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 验证集
filenames_val = tf.data.Dataset.list_files([os.path.join(base_dir, "data", "recall_user_item_act.csv")])
dataset_val = filenames_val.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

val_batch_size = 8
dataset_val = dataset_val.map(parse_function, num_parallel_calls=60)
dataset_val = dataset_val.padded_batch(batch_size=val_batch_size)  # 每1024条数据为一个batch，生成一个新的Datasets
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 训练模型
model = DSSM(
    user_feature_columns,
    item_feature_columns,
    user_dnn_hidden_units=(256, 256, 128),
    item_dnn_hidden_units=(256, 256, 128),
    user_dnn_dropout=(0, 0, 0),
    item_dnn_dropout=(0, 0, 0),
    out_dnn_activation='tanh',
    gamma=1,
    dnn_use_bn=False,
    seed=1024,
    metric='cos')

model.compile(optimizer='adagrad',
              loss={"dssm_out": WeightedBinaryCrossEntropy(), },
              loss_weights=[1.0, ],
              metrics={"dssm_out": [tf.keras.metrics.AUC(name='auc')]}
              )

# log_dir = '/mywork/tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
#                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                          write_graph=True,  # 是否存储网络结构图
#                          write_images=True,  # 是否可视化参数
#                          update_freq='epoch',
#                          embeddings_freq=0,
#                          embeddings_layer_names=None,
#                          embeddings_metadata=None,
#                          profile_batch=40)

total_train_sample = 100
total_test_sample = 100
train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)
test_steps_per_epoch = np.ceil(total_test_sample / val_batch_size).astype(np.int32)
history_loss = model.fit(dataset,
                         epochs=1,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_data=dataset_val,
                         validation_steps=test_steps_per_epoch,
                         verbose=1,
                         # callbacks=[tbCallBack]
                         )

# 保存模型
# 用户塔 item塔定义
user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
# 保存

tf.keras.models.save_model(user_embedding_model, os.path.join(base_dir, "save_model", "dssmUser", "001"))
tf.keras.models.save_model(item_embedding_model, os.path.join(base_dir, "save_model", "dssmIterm", "001"))

# 获取user embedding 及item embedding
user_query = {'client_type': np.array(['1']),
              'all_topic_fav_7': np.array(['1'])}

item_query = {'post_id': np.array(['002']),
              'topic_id': np.array(['2'])}

user_embs = user_embedding_model.predict(user_query)
item_embs = item_embedding_model.predict(item_query)

print(user_embs)
print(item_embs)