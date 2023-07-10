import matplotlib.pyplot as plt
import numpy as npy
import seaborn as sns

from data.DataLoader import Data

data = Data()
data.get_data()

cs = []
scs = []
heat = npy.zeros((10, 100))

for project_id in data.project_info.keys():
    project = data.project_info[project_id]
    c = project["category"]
    sc = project["sub_category"]
    cs.append(c)
    scs.append(sc)
    heat[c - 1][sc - 1] += 1


print("Category Max: %d, Sub Category Max: %d" % (max(cs), max(scs)))
print("Category Min: %d, Sub Category Min: %d" % (min(cs), min(scs)))

plt.hist(cs)
plt.show()
plt.hist(scs)
plt.show()
sns.heatmap(npy.log(heat + 1))
plt.show()
