import os
import sys

input_data=sys.argv[1]
low_k=int(sys.argv[2])
high_k=int(sys.argv[3])
cluster_range=[low_k,high_k]
num_iter=int(sys.argv[4])
kmeans_n_init=int(sys.argv[5])
resampling_prop=float(sys.argv[6])
analysis=sys.argv[7] #specify a static or dynamic analysis
verbose=sys.argv[8]
out_path=os.path.abspath(sys.argv[9])
filename=sys.argv[10]
os.makedirs(out_path, exist_ok=True)

import clustering
import numpy as np

if input_data[-3:]=='npy':
    data=np.load(input_data)

if analysis=='static':
    '''sFC: evaluate the stability of clustering across a range of cluster numbers'''

    BASC=clustering.run_BASC(data, cluster_range, num_iter, kmeans_n_init, resampling_prop, verbose, out_path=out_path, filename=filename)

'''
    cluster_range=[2,10]
    num_iter=100
    kmeans_n_init=100
    resampling_prop=0.8
    verbose=True
    out_path="outputs/output_BASC/"
    filename='test'

    data=np.load('python_data/FC_matrices.npy')
    evaluate_over_range=True
'''


if analysis=='dynamic':
    '''dFC: evaluate the stability of clustering across a range of cluster numbers'''
    DySC=clustering.run_DySC(data, cluster_range, num_iter, kmeans_n_init, resampling_prop, verbose, out_path, filename)
'''
    cluster_range=[2,10]
    num_iter=100
    kmeans_n_init=100
    resampling_prop=0.8
    verbose=True
    out_path="outputs/output_DySC/"
    filename='test'

    data=np.load('python_data/all_subject_windows_normalized.npy')
'''

if analysis=='dFC_windows':
    '''dFC: evaluate the stability of clustering across a range of cluster numbers'''
    dFC_boot_clus=clustering.dFC_bootstrap_cluster(data, cluster_range, num_iter, kmeans_n_init, resampling_prop, verbose, out_path, filename)
'''
    cluster_range=[4,6]
    num_iter=10
    kmeans_n_init=1
    resampling_prop=0.8
    verbose=True
    out_path="outputs/test/"
    filename='test'

    data=np.load('python_data/iso_windows.npy')
'''
if analysis=='dFC_pca':
    subjects_window_pca=clustering.load_obj(input_data)
    matrix_comps=subjects_window_pca.get('matrix_components')

    '''dFC: evaluate the stability of clustering across a range of cluster numbers'''
    dFC_boot_clus=clustering.pca_group_bootstrap_cluster(matrix_comps, cluster_range, num_iter, kmeans_n_init, resampling_prop, verbose, out_path, filename)

