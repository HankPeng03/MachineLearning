#encoding=utf8
from utils import *
from config import *
from tensorflow.keras.models import Model

def DSSM(
        user_feature_columns,
        item_feature_columns,
        user_dnn_hidden_units=(256, 256, 128),
        item_dnn_hidden_units=(256, 256, 128),
        user_dnn_dropout=(0, 0, 0),
        item_dnn_dropout=(0, 0, 0),
        out_dnn_activation='tanh',
        gamma=1.2,
        dnn_use_bn=False,
        seed=1024,
        metric='cos'):
    """
    Instantiates the Deep Structured Semantic Model architecture.
    Args:
        user_feature_columns: A list containing user's features used by the model.
        item_feature_columns: A list containing item's features used by the model.
        user_dnn_hidden_units: tuple,tuple of positive integer , the layer number and units in each layer of user tower
        item_dnn_hidden_units: tuple,tuple of positive integer, the layer number and units in each layer of item tower
        out_dnn_activation: Activation function to use in deep net
        dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        user_dnn_dropout: tuple of float in [0,1), the probability we will drop out a given user tower DNN coordinate.
        item_dnn_dropout: tuple of float in [0,1), the probability we will drop out a given item tower DNN coordinate.
        seed: integer ,to use as random seed.
        gamma: A useful hyperparameter for Similarity layer
        metric: str, "cos" for  cosine
    return: A TF Keras model instance.
    """
    features_columns = user_feature_columns + item_feature_columns
    # 构建 embedding_dict
    embedding_matrix = build_embedding_matrix(features_columns)
    embedding_dict = build_embedding_dict(features_columns, embedding_matrix)

    # user 特征 处理
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns, embedding_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # item 特征 处理
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns, embedding_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

    # user tower
    for i in range(len(user_dnn_hidden_units)):
        if i == len(user_dnn_hidden_units) - 1:
            user_dnn_out = CustomDense(units=user_dnn_hidden_units[i],
                                       dropout_rate=user_dnn_dropout[i],
                                       use_bn=dnn_use_bn,
                                       activation=out_dnn_activation,
                                       name='user_embed_out')(user_dnn_input)
            break
        user_dnn_input = CustomDense(units=user_dnn_hidden_units[i],
                                     dropout_rate=user_dnn_dropout[i],
                                     use_bn=dnn_use_bn,
                                     activation='relu',
                                     name='dnn_user_' + str(i))(user_dnn_input)

    # item tower
    for i in range(len(item_dnn_hidden_units)):
        if i == len(item_dnn_hidden_units) - 1:
            item_dnn_out = CustomDense(units=item_dnn_hidden_units[i],
                                       dropout_rate=item_dnn_dropout[i],
                                       use_bn=dnn_use_bn,
                                       activation=out_dnn_activation,
                                       name='item_embed_out')(item_dnn_input)
            break
        item_dnn_input = CustomDense(units=item_dnn_hidden_units[i],
                                     dropout_rate=item_dnn_dropout[i],
                                     use_bn=dnn_use_bn,
                                     activation='relu',
                                     name='dnn_item_' + str(i))(item_dnn_input)

    score = Similarity(type_sim=metric, gamma=gamma)([user_dnn_out, item_dnn_out])
    output = tf.keras.layers.Activation("sigmoid", name="dssm_out")(score)
    #    score = Multiply()([user_dnn_out, item_dnn_out])
    #    output = Dense(1, activation="sigmoid",name="dssm_out")(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model