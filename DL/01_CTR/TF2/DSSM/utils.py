#encoding=utf8
from tensorflow.keras.layers import Layer
from collections import OrderedDict
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from config import *

import tensorflow as tf

class EncodeMultiEmbedding(Layer):
    def __init__(self, embedding, has_weight=False, **kwargs):
        super(EncodeMultiEmbedding, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.embedding = embedding

    def build(self, input_shape):
        super(EncodeMultiEmbedding, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner='sum')
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner='mean')
        return tf.expand_dims(combiner_embed, 1)

    def get_config(self):
        config = super(EncodeMultiEmbedding, self).get_config()
        config.update({'has_weight': self.has_weight})
        return config


class SparseVocabLayer(Layer):
    def __init__(self, keys, **kwargs):
        super(SparseVocabLayer, self).__init__(**kwargs)
        vals = tf.range(1, len(keys) + 1)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 0)

    def call(self, inputs):
        input_idx = tf.where(tf.not_equal(inputs, ''))
        input_sparse = tf.SparseTensor(input_idx, tf.gather_nd(inputs, input_idx), tf.shape(inputs, out_type=tf.int64))
        return tf.SparseTensor(indices=input_sparse.indices,
                               values=self.table.lookup(input_sparse.values),
                               dense_shape=input_sparse.dense_shape)


# 自定义dnese层含BN， dropout
class CustomDense(Layer):
    def __init__(self, units=32, activation='tanh', dropout_rate=0, use_bn=False, seed=1024, tag_name="dnn", **kwargs):
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.tag_name = tag_name

        super(CustomDense, self).__init__(**kwargs)

    # build方法一般定义Layer需要被训练的参数。
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='random_normal',
                                      trainable=True,
                                      name='kernel_' + self.tag_name)
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='random_normal',
                                    trainable=True,
                                    name='bias_' + self.tag_name)

        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()

        self.dropout_layers = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(self.activation, name=self.activation + '_' + self.tag_name)

        super(CustomDense, self).build(input_shape)  # 相当于设置self.built = True

    # call方法一般定义正向传播运算逻辑，__call__方法调用了它。
    def call(self, inputs, training=None, **kwargs):
        fc = tf.matmul(inputs, self.weight) + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        out_fc = self.activation_layers(fc)

        return out_fc

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法，保存模型不写这部分会报错
    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_bn': self.use_bn,
                       'dropout_rate': self.dropout_rate, 'seed': self.seed, 'name': self.tag_name})
        return config


# cos 相似度计算层
class Similarity(Layer):

    def __init__(self, gamma=1, axis=-1, type_sim='cos', **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type_sim = type_sim
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        if self.type_sim == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)
        cosine_score = tf.reduce_sum(tf.multiply(query, candidate), -1)
        cosine_score = tf.divide(cosine_score, query_norm * candidate_norm + 1e-8)
        cosine_score = tf.clip_by_value(cosine_score, -1, 1.0) * self.gamma
        return tf.expand_dims(cosine_score, 1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type_sim}
        base_config = super(Similarity, self).get_config()
        return base_config.uptate(config)


# 自定损失函数，加权交叉熵损失
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, pos_weight=1.2, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)[:, None]
        ce = ce * (1 - y_true) + self.pos_weight * ce * (y_true)
        #         ce =tf.nn.weighted_cross_entropy_with_logits(
        #             y_true, y_pred, self.pos_weight, name=None
        #         )

        return ce

    def get_config(self, ):
        config = {'pos_weight': self.pos_weight, 'from_logits': self.from_logits, 'name': self.name}
        base_config = super(WeightedBinaryCrossEntropy, self).get_config()
        return base_config.uptate(config)

# 定义输入及共享层帮助函数

def build_input_features(features_columns):
    """
    定义model输入特征
    :param features_columns:
    :return:
    """
    input_features = OrderedDict()
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))
    return input_features


# 构造自定义embedding层matrix
def build_embedding_matrix(features_columns):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                vocab_size = feat_col.voc_size
                embed_dim = feat_col.embed_dim
                if vocab_name not in embedding_matrix:
                    embedding_matrix[vocab_name] = tf.Variable(
                        initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim), mean=0.0,
                                                                 stddev=0.0, dtype=tf.float32), trainable=True,
                        name=vocab_name + '_embed')
    return embedding_matrix



def build_embedding_dict(features_columns, embedding_matrix):
    """
    # 构造自定义 embedding层
    :param features_columns:
    :param embedding_matrix:
    :return: dict, {"col_name": tf.nn.embedding_lookup_sparse}
    """
    embedding_dict = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                embedding_dict[feat_col.name] = EncodeMultiEmbedding(embedding=embedding_matrix[vocab_name],
                                                                     name='EncodeMultiEmb_' + feat_col.name)
    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict):
    """
    将原始输入转化为embedding
    :param features:
    :param features_columns:
    :param embedding_dict:
    :return:
    """
    sparse_embedding_list = []
    dense_value_list = []

    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                keys = DICT_CATEGORICAL[vocab_name]
                _input_sparse = SparseVocabLayer(keys)(features[feat_col.name])

        if isinstance(feat_col, SparseFeat):
            if feat_col.dtype == 'string':
                _embed = embedding_dict[feat_col.name](_input_sparse)
            else:
                _embed = Embedding(feat_col.voc_size + 1, feat_col.embed_dim,
                                   embeddings_regularizer=tf.keras.regularizers.l2(0.5), name='Embed_' + feat_col.name)(
                    features[feat_col.name])
            sparse_embedding_list.append(_embed)
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))

    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """
    flatten并combine
    :param sparse_embedding_list:
    :param dense_value_list:
    :return:
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise Exception("dnn_feature_columns can not be empty list")

def parse_function(example_proto):
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim=',')
    parsed = dict(zip(COL_NAME, item_feats))
    feature_dict = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, SparseFeat):
            feature_dict[feat_col.name] = parsed[feat_col.name]
        else:
            raise Exception("unknown feature_columns....")
    label = parsed['act']
    return feature_dict, label