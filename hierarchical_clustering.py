from scipy import io

sFNC=io.loadmat('mat_files/matrix_sFNC.mat')

'''https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/  '''
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
Z = linkage(X, 'average')


# some setting for this notebook to actually show the graphs inline
# you probably won't need this
%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation



# calculate full dendrogram; heights in the dendogram reflects the distance
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# set cut-off to 50 and plot it in the graph
max_d = 50  # max_d as in max_distance

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()

'''automatically determining # of clusters, in cases of unclear #'''

'''
inconsistency method: inconsistency=(height_of_jump-avg_height)/std_height
Limitation:highly dependent on the depth of the three you assign it
'''
from scipy.cluster.hierarchy import inconsistent
#calculate for depth 5 and last 10 merges; but this metric highly depends on the depth of the tree you calculate the average over
depth = 5
incons = inconsistent(Z, depth)
incons[-10:]

'''
Elbow method
Limitation:assumes that the variance in distribution between each clusters is the same
'''

last = Z[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print "clusters:", k
