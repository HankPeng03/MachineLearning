#encoding=utf8
from collections import namedtuple

"""
字段介绍：

act：为label数据 1:正样本，0：负样本

client_id: 用户id

post_id：物料item id 这里称为post_id

client_type:用户客户端类型

follow_topic_id: 用户关注话题分类id

all_topic_fav_7: 用户画像特征，用户最近7天对话题偏爱度刻画，kv键值对形式

topic_id: 物料所属的话题

read_post_id:用户最近阅读的物料id
"""

base_dir = "/Users/penghao/Workspace/GitLab/MachineLearning/DL/01_CTR/TF2/DSSM"

# 定义参数类型
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'share_embed', 'embed_dim', 'dtype'])

feature_columns = [
                   SparseFeat(name='act', voc_size=3, share_embed=None, embed_dim=8, dtype='float'),
                   SparseFeat(name='post_id', voc_size=3, share_embed=None, embed_dim=8, dtype='string'),
                   SparseFeat(name='all_topic_fav_7', voc_size=3, share_embed=None, embed_dim=8, dtype='string'),
                   SparseFeat(name='client_type', voc_size=3, share_embed=None, embed_dim=8, dtype='string'),
                   SparseFeat(name="topic_id", voc_size=700, share_embed=None, embed_dim=16, dtype='string'),
                   ]

# 用户特征及贴子特征
user_feature_columns_name = ['all_topic_fav_7', 'client_type']
item_feature_columns_name = ["topic_id", "post_id"]
user_feature_columns = [col for col in feature_columns if col.name in user_feature_columns_name]
item_feature_columns = [col for col in feature_columns if col.name in item_feature_columns_name]


# 定义离散特征集合 ，离散特征vocabulary
DICT_CATEGORICAL = {
                    "all_topic_fav_7": ['1', '2'],
                    "topic_id": [str(i) for i in range(0, 700)],
                    "client_type": ['0', '1'],
                    "client_id": ['001', '002'],
                    "post_id": ['001', '002']
                    }

# 构造训练tf.dataset数据
DEFAULT_VALUES = [[0.0], [''], [''], [''], [''], ['']]
COL_NAME = ['act', 'client_id', 'post_id', 'client_type', 'all_topic_fav_7', 'topic_id']