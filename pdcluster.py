import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydpc import Cluster
from pydpc._reference import Cluster as RefCluster

import json
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

samples = 6430
jsonfile = open("address.json","r",encoding='utf-8')

jsonobj = json.load(jsonfile)
rows = jsonobj["features"]

sampleValue = []
count = 0
for each in rows:
    if True:#count < samples
        mid = each["properties"]["OBJECTID"]
        stdAdress = each["properties"]["STAADDRESS"]
        coordinates = each["geometry"]["coordinates"]
        sampleValue.append(coordinates)
        count=count+1
points = np.array(sampleValue)

# npoints = 2000
# mux = 1.6
# muy = 1.6
# points = np.zeros(shape=(npoints, 2), dtype=np.float64)
# points[:, 0] = np.random.randn(npoints) + mux * (-1)**np.random.randint(0, high=2, size=npoints)
# points[:, 1] = np.random.randn(npoints) + muy * (-1)**np.random.randint(0, high=2, size=npoints)
# # draw the data points
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(points[:, 0], points[:, 1], s=40)
# ax.plot([-mux, -mux], [-1.5 * muy, 1.5 * muy], '--', linewidth=2, color="red")
# ax.plot([mux, mux], [-1.5 * muy, 1.5 * muy], '--', linewidth=2, color="red")
# ax.plot([-1.5 * mux,  1.5 * mux], [-muy, -muy], '--', linewidth=2, color="red")
# ax.plot([-1.5 * mux,  1.5 * mux], [muy, muy], '--', linewidth=2, color="red")
# ax.set_xlabel(r"x / a.u.", fontsize=20)
# ax.set_ylabel(r"y / a.u.", fontsize=20)
# ax.tick_params(labelsize=15)
# ax.set_xlim([-7, 7])
# ax.set_ylim([-7, 7])
# ax.set_aspect('equal')
# fig.tight_layout()
# plt.show()

clu = Cluster(points)

clu.assign(50, 150)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(points[:, 0], points[:, 1], s=1)
ax[0].scatter(points[clu.clusters, 0], points[clu.clusters, 1], s=10, c="red")
ax[1].scatter(points[:, 0], points[:, 1], s=1, c=clu.density)
ax[2].scatter(points[:, 0], points[:, 1], s=1, c=clu.membership, cmap = mpl.cm.cool)
# for _ax in ax:
#     _ax.plot([-mux, -mux], [-1.5 * muy, 1.5 * muy], '--', linewidth=2, color="red")
#     _ax.plot([mux, mux], [-1.5 * muy, 1.5 * muy], '--', linewidth=2, color="red")
#     _ax.plot([-1.5 * mux,  1.5 * mux], [-muy, -muy], '--', linewidth=2, color="red")
#     _ax.plot([-1.5 * mux,  1.5 * mux], [muy, muy], '--', linewidth=2, color="red")
#     _ax.set_xlabel(r"x / a.u.", fontsize=20)
#     _ax.set_ylabel(r"y / a.u.", fontsize=20)
#     _ax.tick_params(labelsize=15)
#     _ax.set_xlim([-7, 7])
#     _ax.set_ylim([-7, 7])
#     _ax.set_aspect('equal')
# fig.tight_layout()
plt.show()