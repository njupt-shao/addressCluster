import jieba
from gensim import corpora, models, similarities
import matplotlib.pyplot as mp
import seaborn
from scipy.cluster import hierarchy
import pandas as pd
import numpy as np
import Levenshtein

mp.rcParams['font.sans-serif']=['SimHei'] 

# Tf-idf
def cross(i, cross_dictionary):
    dictionary = cross_dictionary[i]
    feature_cnt = len(dictionary.token2id.keys())
    corpus = [dictionary.doc2bow(text) for text in cross_text[i][1]]
    vector = dictionary.doc2bow(cross_text[i][0])
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[vector]]
    return [value for value in sim]



# Levenshtein distance
def levenshtein_distance(first, second):
    '''
    计算两个字符串之间的L氏编辑距离
    :输入参数 first: 第一个字符串
    :输入参数 second: 第二个字符串
    :返回值: L氏编辑距离
    '''
    if len(first) == 0 or len(second) == 0:
        return len(first) + len(second)
    first_length = len(first) + 1
    second_length = len(second) + 1
    distance_matrix = [range(second_length) for i in range(first_length)] # 初始化矩阵
    for i in range(1, first_length):
        for j in range(1, second_length):
            deletion = distance_matrix[i-1][j] + 1
            insertion = distance_matrix[i][j-1] + 1
            substitution = distance_matrix[i-1][j-1]
            if first[i-1] != second[j-1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[first_length-1][second_length-1]



f = open(r"rawtown.txt",encoding='utf-8')  
rawtowns = []           
line = f.readline()             
while line:
    info=line.split('/')
    rawtowns.append(info[0])
    line = f.readline()
f.close()
length = len(rawtowns)
print (rawtowns)



###############################################################################################################################
cross_text = [([word for word in jieba.cut(i)], [[word for word in jieba.cut(j)] for j in rawtowns if j != i]) for i in rawtowns]
# cross_dictionary = [corpora.Dictionary(i[1]) for i in cross_text]
# sim_matrix = [cross(i, cross_dictionary) for i in range(length)]
# for i in range(len(sim_matrix)):
#     sim_matrix[i].insert(i, 1)

# # visulize
# #seaborn.heatmap(sim_matrix, center=1, annot=True)
# # mp.show()

# sim_matrix = pd.DataFrame(sim_matrix)
# Z = hierarchy.linkage(sim_matrix, method ='average',metric='euclidean')#ward,complete
# hierarchy.dendrogram(Z,labels = rawtowns)
# mp.show()
##################################################################################################################################
dis_mat =np.zeros((len(rawtowns),len(rawtowns)))
print(dis_mat)
i=0
j=0
for iitem in rawtowns:
    for jitm in  rawtowns:
        dis_mat[i][j] = Levenshtein.distance(rawtowns[i],rawtowns[j])
        j = j+1
    i = i+1

print(dis_mat)
