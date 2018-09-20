import jieba
def cut(sentence):
    generator = jieba.cut(sentence)
    return [word for word in generator]
text0 = '多数学院已经放学，但多数学生仍在学习'
text1 = '数学专业的学生多数仍在学院学习'
text2 = '补习数学'
texts = [text0, text1, text2]
length = len(texts)
cross_text = [([word for word in jieba.cut(i)], [[word for word in jieba.cut(j)] for j in texts if j != i]) for i in texts]
# 文本相似度矩阵
from gensim import corpora, models, similarities
cross_dictionary = [corpora.Dictionary(i[1]) for i in cross_text]
def cross(i, cross_dictionary):
    dictionary = cross_dictionary[i]
    feature_cnt = len(dictionary.token2id.keys())
    corpus = [dictionary.doc2bow(text) for text in cross_text[i][1]]
    vector = dictionary.doc2bow(cross_text[i][0])
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[vector]]
    return [value for value in sim]
sim_matrix = [cross(i, cross_dictionary) for i in range(length)]
for i in range(len(sim_matrix)):
    sim_matrix[i].insert(i, 1)
# 可视化
import matplotlib.pyplot as mp, seaborn
seaborn.heatmap(sim_matrix, center=1, annot=True)
mp.show()