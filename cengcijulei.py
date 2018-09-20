import json
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
samples = 5000
jsonfile = open("address.json","r",encoding='utf-8')

jsonobj = json.load(jsonfile)
rows = jsonobj["features"]

sampleValue = []
count = 0
for each in rows:
    if count < samples:
        mid = each["properties"]["OBJECTID"]
        stdAdress = each["properties"]["STAADDRESS"]
        coordinates = each["geometry"]["coordinates"]
        sampleValue.append(coordinates)
        count=count+1

    
sampleValue = pd.core.frame.DataFrame(sampleValue)
mergings = linkage(sampleValue,method='complete')
dendrogram(mergings,leaf_rotation=0,leaf_font_size=10)
plt.show()