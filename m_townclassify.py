import jieba
from gensim import corpora, models, similarities
import matplotlib.pyplot as mp, seaborn

# calculate matrix
def cross(i, cross_dictionary):
    dictionary = cross_dictionary[i]
    feature_cnt = len(dictionary.token2id.keys())
    corpus = [dictionary.doc2bow(text) for text in cross_text[i][1]]
    vector = dictionary.doc2bow(cross_text[i][0])
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[vector]]
    return [value for value in sim]


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
cross_text = [([word for word in jieba.cut(i)], [[word for word in jieba.cut(j)] for j in rawtowns if j != i]) for i in rawtowns]
cross_dictionary = [corpora.Dictionary(i[1]) for i in cross_text]
sim_matrix = [cross(i, cross_dictionary) for i in range(length)]
for i in range(len(sim_matrix)):
    sim_matrix[i].insert(i, 1)

# visulize
seaborn.heatmap(sim_matrix, center=1, annot=True)
mp.show()

