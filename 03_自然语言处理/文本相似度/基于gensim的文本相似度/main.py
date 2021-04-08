# encoding:utf8
import pandas as pd
import numpy as np
import json
import jieba.posseg as pseg
import jieba
from jieba.analyse import textrank
import distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import pdb
# from pyltp import Parser
# from pyltp import NamedEntityRecognizer
from gensim import models, corpora


class model(object):
    def __init__(self, text1, text2, tfidf_model, stop_words):
        self.text1 = text1
        self.text2 = text2
        self.tfidf_model = tfidf_model
        self.stop_words = stop_words

    def get_features(self):
        # 分词
        text1_tokens = self.tokens(self.text1)
        text2_tokens = self.tokens(self.text2)
        # 去除停用词
        text1_tokens = self.drop_skip_word(text1_tokens)
        text2_tokens = self.drop_skip_word(text2_tokens)
        # 编辑距离
        if max(len(self.text1), len(self.text2)) > 0:
            levenshtein_total = self.calculate_levenshtein(self.text1, self.text2) / max(len(self.text1),
                                                                                         len(self.text2))
        else:
            levenshtein_total = 0.0
        ## 筛选名词，计算编辑距离
        text1_n = "".join([w for w, f in text1_tokens if f == 'n'])
        text2_n = "".join([w for w, f in text2_tokens if f == 'n'])
        if max(len(text1_n), len(text2_n)) > 0:
            levenshtein_n = self.calculate_levenshtein(text1_n, text2_n) / max(len(text1_n), len(text2_n))
        else:
            levenshtein_n = 0.0
        ## 筛选动词，计算编辑距离
        text1_v = "".join([w for w, f in text1_tokens if f == 'v'])
        text2_v = "".join([w for w, f in text2_tokens if f == 'v'])
        if max(len(text1_v), len(text2_v)) > 0:
            levenshtein_v = self.calculate_levenshtein(text1_v, text2_v) / max(len(text1_v), len(text2_v))
        else:
            levenshtein_v = 0.0
        ## 关键词编辑距离
        text1_keyword = "".join(self.extract_key_words(self.text1))
        text2_keyword = "".join(self.extract_key_words(self.text2))
        if max(len(text1_keyword), len(text2_keyword)) > 0:
            levenshtein_keyword = self.calculate_levenshtein(text1_keyword, text2_keyword) / max(len(text1_keyword),
                                                                                                 len(text2_keyword))
        else:
            levenshtein_keyword = 0.0
        # 相似度
        text1 = " ".join([w for w, f in text1_tokens])
        text2 = " ".join([w for w, f in text2_tokens])
        if (min(len(text1), len(text2)) == 0) & (max(len(text1), len(text2)) > 0):
            sim_total = 0.0
            sim_n = 0.0
            sim_v = 0.0
            sim_key = 0.0
        else:
            text1_vector = self.tfidf_model.transform([text1]).toarray().flatten()
            text2_vector = self.tfidf_model.transform([text2]).toarray().flatten()
            sim_total = self.cos_sim(text1_vector, text2_vector)
            ## 筛选名词，计算相似度
            text1_n = " ".join([w for w, f in text1_tokens if f == 'n'])
            text2_n = " ".join([w for w, f in text2_tokens if f == 'n'])
            if (min(len(text1_n), len(text2_n)) == 0) & (max(len(text1_n), len(text2_n)) > 0):
                sim_n = 0.0
            elif (len(text1_n) == 0) & (len(text2_n) == 0):
                sim_n = 1.0
            else:
                text1_n_vector = self.tfidf_model.transform([text1_n]).toarray().flatten()
                text2_n_vector = self.tfidf_model.transform([text2_n]).toarray().flatten()
                sim_n = self.cos_sim(text1_n_vector, text2_n_vector)
            ## 筛选动词，计算相似度
            text1_v = " ".join([w for w, f in text1_tokens if f == 'v'])
            text2_v = " ".join([w for w, f in text2_tokens if f == 'v'])
            if (min(len(text1_v), len(text2_v)) == 0) & (max(len(text1_v), len(text2_v)) > 0):
                sim_v = 0.0
            elif (len(text1_v) == 0) & (len(text1_v)) == 0:
                sim_v = 1.0
            else:
                pdb.set_trace()
                text1_v_vector = self.tfidf_model.transform([text1_v]).toarray().flatten()
                text2_v_vector = self.tfidf_model.transform([text2_v]).toarray().flatten()
                sim_v = self.cos_sim(text1_v_vector, text2_v_vector)
            ## 关键词相似度
            text1_keyword = " ".join(self.extract_key_words(self.text1))
            text2_keyword = " ".join(self.extract_key_words(self.text2))
            if (min(len(text1_keyword), len(text2_keyword)) == 0) & (max(len(text1_keyword), len(text2_keyword)) > 0):
                sim_key = 0.0
            elif (len(text1_keyword) == 0) & (len(text2_keyword) == 0):
                sim_key = 1.0
            else:
                text1_key_vector = self.tfidf_model.transform([text1_keyword]).toarray().flatten()
                text2_key_vector = self.tfidf_model.transform([text2_keyword]).toarray().flatten()
                sim_key = self.cos_sim(text1_key_vector, text2_key_vector)

        # 长度差
        diff_len = abs(len(self.text1) - len(self.text2)) / min(len(self.text1), len(self.text2))

        # 共现词比例
        ratio = self.common_word_ratio(text1_tokens, text2_tokens)

        # n-gram距离
        dist_ngram = self.ngram_dist(self.text1, self.text2)

        # 杰卡德系数
        jaccard = self.jaccard_sim(self.text1, self.text2)

        # 最长公共子串
        _, most_common_sub_str_length = self.getNumofCommonSubStr(self.text1, self.text2)
        most_common_sub_str_length_ratio = most_common_sub_str_length / min(len(self.text1), len(self.text2))

        return pd.Series(data={"levenshtein_total": levenshtein_total,
                               "levenshtein_n": levenshtein_n,
                               "levenshtein_v": levenshtein_v,
                               "levenshtein_keyword": levenshtein_keyword,
                               "dist_sim_total": 1.0 - sim_total,
                               "dist_sim_n": 1.0 - sim_n,
                               "dist_sim_v": 1.0 - sim_v,
                               "dist_sim_key": 1.0 - sim_key,
                               "diff_len": diff_len,
                               "dist_ratio": 1.0 - ratio,
                               "dist_ngram": dist_ngram,
                               "dist_jaccard": 1 - jaccard,
                               "dist_most_common_sub_str_length_ratio": 1 - most_common_sub_str_length_ratio})

    def tokens(self, text):
        """
        分词
        :param text:
        :return:
        """
        return [(w.word, w.flag) for w in pseg.lcut(text)]

    def drop_skip_word(self, text_tokens):
        """
        去除停用词
        :param text:
        :return:
        """
        global stop_words
        return [(w, f) for w, f in text_tokens if w not in stop_words]

    def cos_sim(self, vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def calculate_levenshtein(self, text1, text2):
        """
        编辑距离
        :param text1:
        :param text2:
        :return:
        """
        return distance.levenshtein(text1, text2)

    def common_word_ratio(self, text1_tokens, text2_tokens):
        """
        共现词比例
        :param text1_tokens:
        :param text2_tokens:
        :return:
        """
        text1_tokens = set([w for w, f in text1_tokens])
        text2_tokens = set([w for w, f in text2_tokens])
        common_words = text1_tokens.intersection(text2_tokens)
        return len(common_words) / min(len(text1_tokens), len(text2_tokens))

    def ngram_dist(self, text1, text2):
        """
        n-gram距离
        :param text1:
        :param text2:
        :return:
        """
        n = 2

        def get_ngram(str1):
            res = []
            for i in range(len(str1)):
                if i < len(str1) + 1 - n:
                    res.append(str1[i: i + n])
            return res

        text1_ngram = get_ngram(text1)
        text2_ngram = get_ngram(text2)
        text1_ngram_set = set(text1_ngram)
        text2_ngram_set = set(text2_ngram)

        return (len(text1_ngram_set) + len(text2_ngram_set) - 2 * len(
            text1_ngram_set.intersection(text2_ngram_set))) / (len(text1_ngram) + len(text2_ngram))

    def jaccard_sim(self, text1, text2):
        # 杰卡德系数
        s1 = " ".join(text1)
        s2 = " ".join(text2)
        cv = CountVectorizer(tokenizer=lambda x: x.split(" "))
        vectors = cv.fit_transform([s1, s2]).toarray()
        inner1 = np.sum(np.min(vectors, axis=0))  # 交集
        outer1 = np.sum(np.max(vectors, axis=0))  # 并集
        return inner1 / outer1

    def getNumofCommonSubStr(self, text1, text2):
        """
        # 最长公共子串
        :param text1:
        :param text2:
        :return:
        """
        lstr1 = len(text1)
        lstr2 = len(text2)
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        maxNum = 0  # 最长匹配长度
        p = 0  # 匹配的起始位
        for i in range(lstr1):
            for j in range(lstr2):
                if text1[i] == text2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]  # 获取最大匹配长度
                        p = i + 1  # 记录最大匹配长度的终止位置
        return text2[p - maxNum:p], maxNum

    def extract_key_words(self, text, topK=3, allowPOS=['ns', 'n']):
        """
        提取关键词
        :param text:
        :param keynum:
        :return:
        """
        return textrank(text, topK=topK, allowPOS=allowPOS)

    # def text_parser(self, text_tokens):
    #     """
    #     依存分析
    #     :param text_tokens:
    #     :return:
    #     """
    #     parser = Parser()
    #     parser.load(self.par_model_path)
    #     words = [w for w, _ in text_tokens]
    #     postags = [f for _, f in text_tokens]
    #     arcs = parser.parse(words, postags)
    #     parser.release()  # 释放模型
    #     return {arc.head: arc.relation for arc in arcs}

    # def text_ner(self, text_tokens):
    #     recognizer = NamedEntityRecognizer()  # 初始化实例
    #     recognizer.load(self.ner_model_path)  # 加载模型
    #     words = [w for w, _ in text_tokens]
    #     postags = [f for _, f in text_tokens]
    #     netags = recognizer.recognize(words, postags)  # 命名实体识别
    #     recognizer.release()  # 释放模型
    #     return ','.join(netags)


if __name__ == "__main__":
    # 读取数据
    data1 = pd.read_csv("Data/log_data.csv")
    str_list = []
    for i in data1["text"].tolist():
        str_list.extend(str(i).split(";"))
    df = pd.DataFrame(data={"text": str_list})
    df = df.loc[df['text'] != '']  # 去除空白字符串
    df.drop_duplicates(inplace=True)  # 去除重复项
    # 替换错别字，替换近义词
    # 分词并过滤停用词
    with open("Data/stop_words_zh.txt", mode="r", encoding="utf8") as f:
        stop_words = f.readlines()
        stop_words = [w.strip() for w in stop_words]

    # TFIDF模型
    str_list = df['text'].tolist()
    strs_tokens = [" ".join(jieba.lcut(text)) for text in str_list]
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf.fit(strs_tokens)
    with open("tfidf_model.pickle", mode="wb") as f:
        pickle.dump(tfidf, f)

    text1 = "如何进行产品注册"
    # text1 = "电视机家庭影像产品"
    text2 = "电视机家庭影像产品"
    m = model(text1, text2, tfidf_model=tfidf, stop_words=stop_words)
    m.get_features()
    # TFIDF相似度

    # 抽取句子中相同词性的词汇，拼接在一起，并计算其向量化后的距离
    # 抽取关键词，拼接在一起，并计算其向量化后的距离（使用SGRank算法来抽取关键词）

    # 依存分析，将句子根据语法分为几个部分，计算几个部分之间的相似度

    # 根据命名实体识别，提取出不同组的词汇
