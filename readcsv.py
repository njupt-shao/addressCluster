f = open(r"D:\001实验数据\少量数据.csv")             
line = f.readline()             
while line:
    info=line[:-1].split(',')
    print (info)
    line = f.readline()
f.close()