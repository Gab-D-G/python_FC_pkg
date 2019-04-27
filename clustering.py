#!/usr/bin/env python3
import os
from sklearn.cluster import KMeans
import sys
cur=os.path.abspath('')
sys.path.insert(0, cur)

from python_FC_pkg import fc_utils

import numpy as np
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

'''
Evaluate within cluster variability.
'''
def eval_var_clusters(FC_vectors, num_roi, cluster_range, kmeans_n_init):
    #clustering on k=2:10, and evaluate which num fits better according to
    #the mean within cluster variance and average silhouette
    stacked_matrices=fc_utils.stack_matrices(FC_vectors, num_roi)
    within_var=np.empty([len(range(cluster_range[0], cluster_range[1]+1))])
    c=0
    for k in range(cluster_range[0], cluster_range[1]+1):
        print(k)
        kmeans = KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                    n_init=kmeans_n_init, tol=1e-6).fit(stacked_matrices)
        labels=kmeans.labels_
        [cluster_var, mean_var]=fc_utils.within_cluster_var(stacked_matrices, labels, k);
        within_var[c]=mean_var;
        c+=1
    return within_var

def cluster_windows_over_range(all_subject_windows, num_nodes, cluster_range, kmeans_n_init, verbose=True):
    vectored_windows=np.zeros([np.size(all_subject_windows, 0)*np.size(all_subject_windows, 1), int((num_nodes**2-num_nodes)/2)])
    c=0
    for i in range(np.size(all_subject_windows, 0)):
        for j in range(np.size(all_subject_windows, 1)):
            window=all_subject_windows[i,j,:,:]
            vectored_windows[c,:]=fc_utils.unwarp_to_vector(window)
            c+=1
    all_labels=np.empty([len(range(cluster_range[0], cluster_range[1]+1)), np.size(vectored_windows, 0)])
    for k in range(cluster_range[0], cluster_range[1]+1):
        kmeans = KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                    n_init=kmeans_n_init, tol=1e-6).fit(vectored_windows)
        all_labels[k-cluster_range[0],:]=kmeans.labels_
        if verbose:
            print("Evaluated "+str(k)+" clusters.")
    return all_labels, vectored_windows

#take num_window*nodes*nodes input with corresponding labels
def get_window_states(windows, labels):
    num_roi=int(np.size(windows, 1))
    k=int(np.max(labels)+1)
    all_states=np.zeros([k,num_roi, num_roi])
    for cluster in range(k):
        indices=labels==cluster
        state=np.zeros([num_roi, num_roi])
        for i in range(len(indices)):
            if indices[i]:
                state+=windows.take(i, 0)
        all_states[cluster,:,:]=state/np.sum(indices)
    return all_states


'''
Will evaluate the cosine similarity (Ben-Hur, 2002) from k_means clustering iteratively over
two random halves of the sample, over a given range of cluster #.
Clustering is applied over n_subject*FC_vector matrices.
'''

def evaluate_cosine_similarity(FC_vectors, num_roi, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, verbose=True):
    #cluster_range should be either a list or array [min_k, max_k]
    from .Stability_Utils import countCommonEdge
    import random
    num_sample=int(np.size(FC_vectors,0))
    k_cosines=np.empty([len(range(cluster_range[0], cluster_range[1]+1)), num_iter])
    c=0
    for k in range(cluster_range[0], cluster_range[1]+1):
        cosines=np.empty([num_iter])
        for i in range(0,num_iter):
            '''
            ramdomly split samples into two groups, each of which containing a
            defined proportion of the subjects commonly included in the two
            groups (default=0.8=80%), and the remaining subjects are
            non-overlaping (e.g. 20%).
            Then evaluate cosine similarity between the two
            '''
            random_order=random.sample(range(num_sample), num_sample)
            group1=FC_vectors[random_order[:int(num_sample*resampling_prop)], :]
            group2=FC_vectors[random_order[int(num_sample-(num_sample*resampling_prop)):], :]
            stacked1=fc_utils.stack_matrices(group1, num_roi)
            stacked2=fc_utils.stack_matrices(group2, num_roi)

            cluster1=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init, tol=1e-6).fit(stacked1)
            cluster2=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init, tol=1e-6).fit(stacked2)
            '''
            evaluate the cosine similarity as defined in Ben-Hur 2002
            '''
            cos_sim=countCommonEdge(cluster1.labels_, cluster2.labels_, verbose=False)/np.sqrt(countCommonEdge(cluster1.labels_, cluster1.labels_, verbose=False)*countCommonEdge(cluster2.labels_, cluster2.labels_, verbose=False))
            cosines[i]=cos_sim
            if verbose:
                print("Eval for "+str(k)+" clusters: iter #"+str(i+1))
        k_cosines[c,:]=cosines
        c+=1
    return k_cosines

'''
Takes as inputs subjects' PCA components of their time windows, for each K
group level clusters, take K first components from subjects, bootstrap cluster
at group level then cluster on final stability matrix + evaluate stability.
'''
def pca_group_bootstrap_cluster(all_subject_components, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, verbose=True, out_path='', filename='dFC_pca_boot_cluster'):
    #input components are in matrix form, so have a n_subs x n_comps x n_roi x n_roi array
    np.seterr(all='raise')
    import warnings
    warnings.filterwarnings("error")
    import random
    os.makedirs(out_path+'/dFC_pca_boot_cluster/tmp', exist_ok=True)

    num_subject=np.size(all_subject_components, 0)
    num_roi=np.size(all_subject_components, 2)
    num_clusters=len(range(cluster_range[0], cluster_range[1]+1))

    all_subsample_labels=[]
    all_stability_matrices=[]

    #try to recover previous runtime to avoid repeating
    try:
        temp_subsamples=load_obj(out_path+'/dFC_pca_boot_cluster/tmp/temp_subsamples')
        low_bound=temp_subsamples.get('k')+1
        all_subsample_labels=temp_subsamples.get('subsample_labels')
        all_stability_matrices=temp_subsamples.get('stability_matrices')
        print('Loading previous subsample cluster '+str(low_bound)+'.')
    except:
        low_bound=cluster_range[0]
        print('No previous run.')
    
    for k in range(low_bound, cluster_range[1]+1):
        #will take k first comps from subjects
        num_comps=k
        k_subsample_labels=np.zeros([num_iter,num_subject*num_comps])
        error=True
        while(error):
            try:
                for i in range(num_iter):
                    #generates a random selection of resampling_prop % of the sample
                    random_subject=random.sample(range(num_subject), num_subject)
                    random_roi=random.sample(range(num_roi), num_roi)
                    temp_subsample_matrices=all_subject_components.take(random_subject[:int(num_subject*resampling_prop)], axis=0)
                    #HERE WE TAKE PART OF THE ROIS, BUT ALSO TAKE THE K FIRST COMPONENTS
                    subsample_matrices=temp_subsample_matrices[:,:num_comps,:,:].take(random_roi[:int(num_roi*resampling_prop)], axis=2)
                    #then stack the windows into vectors
                    stacked_subsamples=np.zeros([np.size(subsample_matrices,0)*np.size(subsample_matrices,1), np.size(subsample_matrices,2)*np.size(subsample_matrices,3)])
                    c=0
                    for sub in range(int(num_subject*resampling_prop)):
                        for comp in range(num_comps):
                            vertices=np.resize(subsample_matrices[sub,comp,:,:], np.size(subsample_matrices,2)*np.size(subsample_matrices,3))
                            stacked_subsamples[c,:]=vertices
                            c+=1
        
                    kmeans=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                                        n_init=kmeans_n_init, tol=1e-6).fit(stacked_subsamples)
                    labels=kmeans.labels_
        
                    #reorder the random_order
                    temp_sort_order=np.argsort(random_subject[:int(num_subject*resampling_prop)]) #obtain the indices for a sorted array
                    sort_order=np.zeros([np.size(labels)])
                    for j in range(len(temp_sort_order)):
                        start=temp_sort_order[j]*num_comps
                        sub_order=np.arange(start,start+num_comps, 1)
                        sort_order[j*num_comps:((j*num_comps)+num_comps)]=sub_order
        
                    temp_sorted_order=np.sort(random_subject[:int(num_subject*resampling_prop)])
                    sorted_order=np.zeros([np.size(labels)])
                    for j in range(len(temp_sorted_order)):
                        start=temp_sorted_order[j]*num_comps
                        sub_order=np.arange(start,start+num_comps, 1)
                        sorted_order[j*num_comps:((j*num_comps)+num_comps)]=sub_order        

                    reordered_labels=fc_utils.reorder(labels, sort_order.astype(int))
                    c=0
                    while(sorted_order[0]!=c):
                        k_subsample_labels[i,c]=-1
                        c+=1
                    for j in range(0, len(sorted_order)):
                        while(sorted_order[j]!=c):
                            k_subsample_labels[i,c]=-1
                            c+=1
                        k_subsample_labels[i,c]=int(reordered_labels[j])
                        c+=1
        
                    while(c<num_subject*num_comps):
                        k_subsample_labels[i,c]=-1
                        c+=1
                    if verbose:
                        print("Eval for "+str(k)+" clusters: boostrap iter #"+str(i+1))
        
                stb=stability_matrix(k_subsample_labels[:,:])
                all_stability_matrices.append(stb)
                all_subsample_labels.append(k_subsample_labels)
                temp_subsamples={'subsample_labels':all_subsample_labels, 'stability_matrices':all_stability_matrices,'current_iteration':i, 'k':k}
                save_obj(temp_subsamples, out_path+'/dFC_pca_boot_cluster/tmp/temp_subsamples')
                error=False
            except FloatingPointError:
                error=True
                print('A division by 0 has occurred. Retry the bootstrap.')

    stability_evaluation=np.zeros([num_clusters])
    all_labels=[]
    eigenconnectivities=[]
    for k in range(cluster_range[0], cluster_range[1]+1):
        if verbose:
            print("Evaluating consensus clustering for k="+str(k))
        stb=all_stability_matrices[k-cluster_range[0]][:,:]
        kmeans=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init*10, tol=1e-6).fit(stb)
        labels=kmeans.labels_
        all_labels.append(labels)
        scores=stability_scores(stb, labels)
        stability_crit=stability_criterion(scores, labels)
        stability_evaluation[k-cluster_range[0]]=stability_crit

        stacked_comps=np.zeros([num_subject*k, num_roi, num_roi])
        c=0
        for sub in range(int(num_subject)):
            for comp in range(k):
                stacked_comps[c,:,:]=all_subject_components[sub,comp,:,:]
                c+=1
    
        group_eigenconnectivities=np.zeros([k, num_roi,num_roi])
        for state in range(k):
            indices=labels==state
            group_state=np.zeros([num_roi, num_roi])
            for i in range(len(indices)):
                if indices[i]:
                    group_state+=stacked_comps.take(i, 0)
            group_eigenconnectivities[state,:,:]=group_state/np.sum(indices)
        eigenconnectivities.append(group_eigenconnectivities)
    final_output={'consensus_dynamic_eigenconnectivities':eigenconnectivities, 'stability_evaluation': stability_evaluation, 'all_labels':all_labels}
    save_obj(final_output, out_path+'/dFC_pca_boot_cluster/'+filename)
    os.remove(out_path+'/dFC_pca_boot_cluster/tmp/temp_subsamples.pkl')
    return final_output

    
    
'''
This function takes the subjects' windows [n_subject,n_windows,num_roi,num_roi],
bootstrap part of the rois and part of the subjects, stack the subsample into a
single matrix of [n_subject*n_windows,num_edges],cluster the windows and store
the labels. After bootstrapping, give a resulting stability matrix for the given
number of clusters, cluster it with the same K and derive the states, evaluate
the stability, store the stability, the states and the labels.
'''

def dFC_bootstrap_cluster(all_subject_timewindows, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, verbose=True, out_path='', filename='dFC_boot_cluster'):
    np.seterr(all='raise')
    import warnings
    import random
    warnings.filterwarnings("error")
    os.makedirs(out_path+'/dFC_boot_cluster/tmp', exist_ok=True)

    num_subject=np.size(all_subject_timewindows, 0)
    num_windows=num_subject*np.size(all_subject_timewindows,1)
    num_roi=np.size(all_subject_timewindows[0,0,:,:], 0)
    num_clusters=len(range(cluster_range[0], cluster_range[1]+1))

    all_subsample_labels=np.empty([num_clusters, num_iter, num_windows])
    all_stability_matrices=np.empty([num_clusters, num_windows,num_windows])

    #try to recover previous runtime to avoid repeating
    try:
        temp_subsamples=load_obj(out_path+'/dFC_boot_cluster/tmp/temp_subsamples')
        low_bound=temp_subsamples.get('k')+1
        all_subsample_labels=temp_subsamples.get('subsample_labels')
        all_stability_matrices=temp_subsamples.get('stability_matrices')
        print('Loading previous subsample cluster '+str(low_bound)+'.')
    except:
        low_bound=cluster_range[0]
        print('No previous run.')

    for k in range(low_bound, cluster_range[1]+1):

        error=True
        while(error):
            try:
                for i in range(num_iter):
                    #generates a random selection of resampling_prop % of the sample
                    random_subject=random.sample(range(num_subject), num_subject)
                    random_roi=random.sample(range(num_roi), num_roi)
                    temp_subsample_matrices=all_subject_timewindows.take(random_subject[:int(num_subject*resampling_prop)], axis=0)
                    subsample_matrices=temp_subsample_matrices.take(random_roi[:int(num_roi*resampling_prop)], axis=2)
                    #then stack the windows into vectors
                    stacked_subsamples=np.zeros([np.size(subsample_matrices,0)*np.size(subsample_matrices,1), np.size(subsample_matrices,2)*np.size(subsample_matrices,3)])
                    c=0
                    for sub in range(int(num_subject*resampling_prop)):
                        for window in range(np.size(subsample_matrices,1)):
                            vertices=np.resize(subsample_matrices[sub,window,:,:], np.size(subsample_matrices,2)*np.size(subsample_matrices,3))
                            stacked_subsamples[c,:]=vertices
                            c+=1

                    kmeans=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                                        n_init=kmeans_n_init, tol=1e-6).fit(stacked_subsamples)
                    labels=kmeans.labels_

                    #reorder the random_order
                    temp_sort_order=np.argsort(random_subject[:int(num_subject*resampling_prop)]) #obtain the indices for a sorted array
                    sort_order=np.zeros([np.size(labels)])
                    for j in range(len(temp_sort_order)):
                        start=temp_sort_order[j]*np.size(all_subject_timewindows, 1)
                        sub_order=np.arange(start,start+np.size(all_subject_timewindows, 1), 1)
                        sort_order[j*np.size(all_subject_timewindows, 1):((j*np.size(all_subject_timewindows, 1))+np.size(all_subject_timewindows, 1))]=sub_order

                    temp_sorted_order=np.sort(random_subject[:int(num_subject*resampling_prop)])
                    sorted_order=np.zeros([np.size(labels)])
                    for j in range(len(temp_sorted_order)):
                        start=temp_sorted_order[j]*np.size(all_subject_timewindows, 1)
                        sub_order=np.arange(start,start+np.size(all_subject_timewindows, 1), 1)
                        sorted_order[j*np.size(all_subject_timewindows, 1):((j*np.size(all_subject_timewindows, 1))+np.size(all_subject_timewindows, 1))]=sub_order

                    reordered_labels=fc_utils.reorder(labels, sort_order.astype(int))
                    c=0
                    while(sorted_order[0]!=c):
                        all_subsample_labels[k-cluster_range[0],i,c]=-1
                        c+=1
                    for j in range(0, len(sorted_order)):
                        while(sorted_order[j]!=c):
                            all_subsample_labels[k-cluster_range[0],i,c]=-1
                            c+=1
                        all_subsample_labels[k-cluster_range[0],i,c]=int(reordered_labels[j])
                        c+=1

                    while(c<num_windows):
                        all_subsample_labels[k-cluster_range[0],i,c]=-1
                        c+=1
                    if verbose:
                        print("Eval for "+str(k)+" clusters: boostrap iter #"+str(i+1))
                stb=stability_matrix(all_subsample_labels[k-cluster_range[0],:,:])
                all_stability_matrices[k-cluster_range[0],:,:]=stb
                temp_subsamples={'subsample_labels':all_subsample_labels, 'stability_matrices':all_stability_matrices,'current_iteration':i, 'k':k}
                save_obj(temp_subsamples, out_path+'/dFC_boot_cluster/tmp/temp_subsamples')
                error=False
            except FloatingPointError:
                error=True
                print('A division by 0 has occurred. Retry the bootstrap.')

    stacked_windows=np.zeros([np.size(all_subject_timewindows,0)*np.size(all_subject_timewindows,1), num_roi, num_roi])
    c=0
    for sub in range(int(num_subject)):
        for window in range(np.size(all_subject_timewindows,1)):
            temp_window=all_subject_timewindows[sub,window,:,:]
            stacked_windows[c,:,:]=temp_window
            c+=1

    stability_evaluation=np.zeros([num_clusters])
    all_labels=np.zeros([num_clusters, num_windows])
    all_group_states=[]
    for k in range(cluster_range[0], cluster_range[1]+1):
        if verbose:
            print("Evaluating consensus clustering for k="+str(k))
        stb=all_stability_matrices[k-cluster_range[0],:,:]
        kmeans=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init*10, tol=1e-6).fit(stb)
        labels=kmeans.labels_
        all_labels[k-cluster_range[0],:]=(labels)
        scores=stability_scores(stb, labels)
        stability_crit=stability_criterion(scores, labels)
        stability_evaluation[k-cluster_range[0]]=stability_crit

        #there will be I*L solutions for M states of size num_roi*num_roi
        group_states=np.zeros([k, num_roi,num_roi])
        for state in range(k):
            indices=labels==state
            group_state=np.zeros([num_roi, num_roi])
            for i in range(len(indices)):
                if indices[i]:
                    group_state+=stacked_windows.take(i, 0)
            group_states[state,:,:]=group_state/np.sum(indices)
        all_group_states.append(group_states)
    final_output={'consensus_dynamic_states':all_group_states, 'stability_evaluation': stability_evaluation, 'all_labels':all_labels}
    save_obj(final_output, out_path+'/dFC_boot_cluster/'+filename)
    os.remove(out_path+'/dFC_boot_cluster/tmp/temp_subsamples.pkl')
    return final_output



'''
This function takes windows and runs DySC.
'''
def run_DySC(all_subject_timewindows, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, verbose=True, out_path='', filename='DySC'):
    np.seterr(all='raise')
    import warnings
    warnings.filterwarnings("error")
    os.makedirs(out_path+'/DySC/tmp', exist_ok=True)

    num_subject=np.size(all_subject_timewindows, 0)
    num_roi=np.size(all_subject_timewindows[0,0,:,:], 0)
    num_clusters=len(range(cluster_range[0], cluster_range[1]+1))

    #if the I_DySC and G_DySC k number are optimized, evaluate over a range
    #of between [70%k, 130%k]
    overall_optimize_range=[round(0.7*cluster_range[0]), round(1.3*cluster_range[1])]
    if overall_optimize_range[0]<2:
        overall_optimize_range[0]=2

    all_states=[]
    #try to recover previous runtime to avoid repeating
    previous_run=False
    try:
        temp_DySC=load_obj(out_path+'/DySC/tmp/temp_individual_DySC')
        previous_run=True
        print('LOADING PREVIOUS RUN.')
    except:
        print('No previous run.')

    if previous_run:
        all_states=temp_DySC.get('all_states')
        low_bound=temp_DySC.get('current_iteration')+1

    else:
        low_bound=overall_optimize_range[0]

    optimize_range=overall_optimize_range
    length_optimize_range=len(range(optimize_range[0], optimize_range[1]+1))
    #individual level DySC
    for I in range(low_bound, overall_optimize_range[1]+1):
        I_states=np.zeros([num_subject*I, num_roi, num_roi])
        #bootstrap cluster at the individual matrix level
        for sub in range(num_subject):
            error=True
            while(error):
                try:
                    if verbose:
                        print("DySC for subject #"+str(sub))
                    subsample_labels=bootstrap_cluster(all_subject_timewindows[sub,:,:,:], boot_axis=1, cluster_axis=0, num_iter=num_iter, kmeans_k=I, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="I_DySC ")
                    stb=stability_matrix(subsample_labels)
                    kmeans=KMeans(n_clusters=I, random_state=0, #set random state to a deterministic initialization
                                        n_init=kmeans_n_init, tol=1e-6).fit(stb)
                    labels=kmeans.labels_
                    states=np.zeros([I, num_roi, num_roi])
                    for c in range(I):
                        indices=c==labels
                        state=np.zeros([num_roi, num_roi])
                        num=0
                        for i in range(len(indices)):
                            if indices[i]:
                                state+=all_subject_timewindows[sub,i,:,:]
                                num+=1
                        states[c,:,:]=state/num
                    I_states[sub*I:(sub+1)*I,:,:]=states
                    error=False
                except FloatingPointError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
        all_states.append(I_states)
        temp_DySC={'all_states':all_states, 'current_iteration':I}
        save_obj(temp_DySC, out_path+'/DySC/tmp/temp_individual_DySC')


    #group level BASC
    group_stability_matrices=[]
    #try to recover previous runtime to avoid repeating
    previous_run=False
    try:
        temp_DySC=load_obj(out_path+'/DySC/tmp/temp_group_DySC')
        previous_run=True
        print('LOADING PREVIOUS RUN.')
    except:
        print('No previous run.')

    if previous_run:
        group_stability_matrices=temp_DySC.get('group_stability_matrices')
        low_bound=temp_DySC.get('current_iteration')+1

    else:
        low_bound=overall_optimize_range[0]

    for L in range(low_bound, overall_optimize_range[1]+1):
        L_stability_matrices=[]
        #bootstrap cluster at the group stability matrix level
        for I in range(optimize_range[0], optimize_range[1]+1):
            I_states=all_states[I-optimize_range[0]]
            error=True
            while(error):
                try:
                    if verbose:
                        print("G_DySC for cluster #"+str(I))
                    group_subsample_labels=bootstrap_cluster(I_states, boot_axis=0, cluster_axis=0, num_iter=num_iter, kmeans_k=L, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="G_DySC ")
                    stb=stability_matrix(group_subsample_labels)
                    L_stability_matrices.append(stb)
                    error=False
                except FloatingPointError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
                except ValueError:
                    error=False
                    L_stability_matrices.append(None)
                    print('# of states probably insufficient for '+str(L)+' group clusters with '+str(int(np.size(I_states,0)*0.8))+' states')
        group_stability_matrices.append(L_stability_matrices)
        temp_DySC={'group_stability_matrices':group_stability_matrices, 'current_iteration':L}
        save_obj(temp_DySC, out_path+'/DySC/tmp/temp_group_DySC')


    optimize_ranges=[] #keep track of the ranges of cluster num over which it was evaluated
    overall_stability_evaluation=[]
    optimal_clustering_solutions=np.zeros([num_clusters, 3])
    all_labels=[]
    all_group_states=[]

    #final consensus cluster evaluation
    for M in range(cluster_range[0], cluster_range[1]+1):
        #if the I_DySC and G_DySC k number are optimized, evaluate over a range
        #of between [70%k, 130%k]
        optimize_range=[round(0.7*M), round(1.3*M)]
        if optimize_range[0]<2:
            optimize_range[0]=2
        optimize_ranges.append(optimize_range)

        length_optimize_range=len(range(optimize_range[0], optimize_range[1]+1))

        M_labels=[]
        M_stability_evaluation=np.zeros([length_optimize_range, length_optimize_range]) #I,L solutions
        for L in range(optimize_range[0], optimize_range[1]+1):
            L_labels=[]
            #bootstrap cluster at the group stability matrix level
            for I in range(optimize_range[0], optimize_range[1]+1):
                if verbose:
                    print("Evaluating consensus clustering for I="+str(I)+" & L="+str(L))
                stb=group_stability_matrices[L-optimize_range[0]][I-optimize_range[0]]
                if type(stb)==type(None):
                    L_labels.append(None)
                    M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=None
                else:
                    try:
                        kmeans=KMeans(n_clusters=M, random_state=0, #set random state to a deterministic initialization
                                            n_init=kmeans_n_init, tol=1e-6).fit(stb)
                        labels=kmeans.labels_
                        L_labels.append(labels)
                        scores=stability_scores(stb, labels)
                        stability_crit=stability_criterion(scores, labels)
                        M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=stability_crit
                    except Warning:
                        print('Probably obtained insufficient # I_states with L='+str(L)+' and I='+str(I))
                        L_labels.append(None)
                        M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=None
                    except ValueError:
                        print('Probably obtained insufficient # I_states with L='+str(L)+' and I='+str(I))
                        L_labels.append(None)
                        M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=None
            M_labels.append(L_labels)

        overall_stability_evaluation.append(M_stability_evaluation)
        all_labels.append(M_labels)

        #now finding the optimal clustering solution for each M
        optimal_stab=M_stability_evaluation[0,0]
        I_optimal=optimize_range[0]
        L_optimal=optimize_range[0]
        for I in range(optimize_range[0], optimize_range[1]+1):
            for L in range(optimize_range[0], optimize_range[1]+1):
                if M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]>optimal_stab:
                    if M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]==None:
                        continue
                    optimal_stab=M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]
                    I_optimal=I
                    L_optimal=L

        optimal_clustering_solutions[(M-cluster_range[0]),:]=np.array([optimal_stab, I_optimal, L_optimal])

        #there will be I*L solutions for M states of size num_roi*num_roi
        group_states=np.zeros([length_optimize_range, length_optimize_range, M,num_roi,num_roi])
        for L in range(optimize_range[0], optimize_range[1]+1):
            for I in range(optimize_range[0], optimize_range[1]+1):
                for state in range(M):
                    if type(M_labels[L-optimize_range[0]][I-optimize_range[0]])==type(None):
                        group_states[I-optimize_range[0],L-optimize_range[0],state,:,:]=None
                        continue
                    labels=M_labels[L-optimize_range[0]][I-optimize_range[0]]
                    indices=labels==state
                    group_state=np.zeros([num_roi, num_roi])
                    for i in range(len(indices)):
                        if indices[i]:
                            group_state+=all_states[I-optimize_range[0]].take(i, 0)
                    group_states[I-optimize_range[0],L-optimize_range[0],state,:,:]=group_state/np.sum(indices)
        all_group_states.append(group_states)

    final_output={'consensus_dynamic_states':all_group_states,'optimal_clustering_solutions':optimal_clustering_solutions, 'overall_stability_evaluation': overall_stability_evaluation, 'optimize_ranges':optimize_ranges,'group_stability_matrices':group_stability_matrices, 'all_states':all_states, 'all_labels':all_labels}
    save_obj(final_output, out_path+'/DySC/'+filename)
    os.remove(out_path+'/DySC/tmp/temp_group_DySC.pkl')
    os.remove(out_path+'/DySC/tmp/temp_individual_DySC.pkl')
    return final_output



'''
Here I introduce Dynamic State Consensus (DySC). This is essentially an adaptation
of the BASC algorithm to find consensus solutions to the clustering of dynamic
windows. It includes both an individual component (I_DySC) and a group
component (G_DySC). I_DySC allows to find weight maps of the states within
windows of an individual, in the following steps:
    1-Bootstrap nodes in the window matrices + cluster subsamples
        *the bootstrapping of nodes will favor finding states which are global
        and not confined within certain regions
    2-derive stability matrix from the subsample solutions + cluster
    3-Evaluate stability criterion from 2) over a range of k, which will be dependent
        on the final number of states which is optimized at the group level
    4-Select the best clustering solution, and output the weighted maps of the
        clustering solutions, resulting in N states.
***will need to make some trials and errors on your data to see if stability
criterion always end taking only the lowest number, which would means that
it doesn't work

G_DySC will take a set of state maps from individual subjects, and find a
consensus solution to these states through clustering:
    1-Takes as inputs sliding window matrices from subjects, and compute
        the I_DySC for each subject to obtain a large set of states from the
        whole sample.
    2-Bootstrap through the states, regardless of the subject + cluster
        the subsample states
        *since not the same states will be clustered across subsamples,
        a None label will be assigned to the output partition to discounted
        these in the computation of the final stability matrix
    3-Evaluate the stability matrix from the subsamples + cluster
    4-Evaluate the stability criterion over a range of k, to select the number
        of states which best fit the sample. The function returns the evaluation
        of the stability criterion and the resulting average maps of the states
        for each number of clusters.

'''

def I_DySC(individual_windows, num_iter, cluster_range, kmeans_n_init, resampling_prop=0.8, verbose=True):
    #takes as input n_windows*nodes*nodes
    num_roi=np.size(individual_windows, 1)
    num_vertices=int(np.size(individual_windows, 0))
    #iterate over a range of k
    num_clusters=len(range(cluster_range[0], cluster_range[1]))+1


    if num_iter<100:
        print('WITH A SMALL NUMBER OF BOOTSTRAPS, THERE IS A HIGH CHANCE OF HAVING A DIVISION BY 0 IN THE STABILITY MATRIX.')


    individual_subsample_labels=np.zeros([num_clusters, num_vertices])
    for I in range(cluster_range[0], cluster_range[1]+1):
        subsample_labels=bootstrap_cluster(individual_windows, boot_axis=1, cluster_axis=0, num_iter=num_iter, kmeans_k=I, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="I_DySC ")
        stb=stability_matrix(subsample_labels)
        kmeans=KMeans(n_clusters=I, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init, tol=1e-6).fit(stb)
        individual_subsample_labels[I-cluster_range[0],:]=kmeans.labels_

    all_states=[]
    for I in range(cluster_range[0], cluster_range[1]+1):
        states=np.zeros([I, num_roi, num_roi])
        for c in range(I):
            indices=c==individual_subsample_labels[I-cluster_range[0], :]
            state=np.zeros([num_roi, num_roi])
            num=0
            for i in range(len(indices)):
                if indices[i]:
                    state+=individual_windows[i,:,:]
                    num+=1
            states[c,:,:]=state/num
        all_states.append(states)
    return all_states, individual_subsample_labels

'''
This function takes n FC_matrices as inputs, runs multilevel bootstraping and
clustering, over a range of clusters M, as defined in the BASC algorithm.
'''
def run_BASC(FC_matrices, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, verbose=True, out_path='', filename='BASC'):
    np.seterr(all='raise')

    os.makedirs(out_path+'/BASC/tmp', exist_ok=True)

    num_subject=np.size(FC_matrices, 0)
    num_roi=np.size(FC_matrices, 1)
    num_clusters=len(range(cluster_range[0], cluster_range[1]+1))

    #if the I_DySC and G_DySC k number are optimized, evaluate over a range
    #of between [70%k, 130%k]
    overall_optimize_range=[round(0.7*cluster_range[0]), round(1.3*cluster_range[1])]
    if overall_optimize_range[0]<2:
        overall_optimize_range[0]=2

    length_optimize_range=len(range(overall_optimize_range[0], overall_optimize_range[1]+1))
    individual_stability_matrices=np.empty([length_optimize_range,num_subject, num_roi, num_roi])

    #try to recover previous runtime to avoid repeating
    previous_run=False
    try:
        temp_BASC=load_obj(out_path+'/BASC/tmp/temp_individual_BASC')
        previous_run=True
        print('LOADING PREVIOUS RUN.')
    except:
        print('No previous run.')

    if previous_run:
        individual_stability_matrices=temp_BASC.get('individual_stability_matrices')
        low_bound=temp_BASC.get('current_iteration')+1

    else:
        low_bound=overall_optimize_range[0]

    optimize_range=overall_optimize_range
    length_optimize_range=len(range(optimize_range[0], optimize_range[1]+1))
    #individual level BASC
    for I in range(low_bound, overall_optimize_range[1]+1):
        #bootstrap cluster at the individual matrix level
        for sub in range(num_subject):
            error=True
            while(error):
                try:
                    if verbose:
                        print("BASC for subject #"+str(sub))
                    subsample_labels=bootstrap_cluster(FC_matrices[sub,:,:], boot_axis=0, cluster_axis=0, num_iter=num_iter, kmeans_k=I, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="BASC Subject"+str(sub))
                    #derive individual stability matrices
                    stb=stability_matrix(subsample_labels)
                    individual_stability_matrices[(I-optimize_range[0]),sub,:,:]=stb

                    error=False
                except FloatingPointError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
        temp_BASC={'individual_stability_matrices':individual_stability_matrices, 'current_iteration':I}
        save_obj(temp_BASC, out_path+'/BASC/tmp/temp_individual_BASC')


    #group level BASC
    L_stability_matrices=np.zeros([length_optimize_range, length_optimize_range, num_roi,num_roi])
    #try to recover previous runtime to avoid repeating
    previous_run=False
    try:
        temp_BASC=load_obj(out_path+'/BASC/tmp/temp_group_BASC')
        previous_run=True
        print('LOADING PREVIOUS RUN.')
    except:
        print('No previous run.')

    if previous_run:
        L_stability_matrices=temp_BASC.get('group_stability_matrices')
        low_bound=temp_BASC.get('current_iteration')+1

    else:
        low_bound=overall_optimize_range[0]
    for L in range(low_bound, overall_optimize_range[1]+1):
        #bootstrap cluster at the group stability matrix level
        for I in range(optimize_range[0], optimize_range[1]+1):
            I_stb=individual_stability_matrices[I-optimize_range[0],:,:,:]
            error=True
            while(error):
                try:
                    group_subsample_labels=bootstrap_cluster(I_stb, boot_axis=0, cluster_axis=1, num_iter=num_iter, kmeans_k=L, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="BASC Group ")
                    stb=stability_matrix(group_subsample_labels)
                    L_stability_matrices[L-optimize_range[0],I-optimize_range[0],:,:]=stb
                    error=False
                except FloatingPointError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
        temp_BASC={'group_stability_matrices':L_stability_matrices, 'current_iteration':L}
        save_obj(temp_BASC, out_path+'/BASC/tmp/temp_group_BASC')


    optimize_ranges=[] #keep track of the ranges of cluster num over which it was evaluated
    overall_stability_evaluation=[]
    optimal_clustering_solutions=np.zeros([num_clusters, 3])
    all_labels=[]

    #final consensus cluster evaluation
    for M in range(cluster_range[0], cluster_range[1]+1):
        #if the I_DySC and G_DySC k number are optimized, evaluate over a range
        #of between [70%k, 130%k]
        optimize_range=[round(0.7*M), round(1.3*M)]
        if optimize_range[0]<2:
            optimize_range[0]=2
        optimize_ranges.append(optimize_range)

        length_optimize_range=len(range(optimize_range[0], optimize_range[1]+1))

        M_labels=np.zeros([length_optimize_range, length_optimize_range, num_roi])
        M_stability_evaluation=np.zeros([length_optimize_range, length_optimize_range]) #I,L solutions
        for L in range(optimize_range[0], optimize_range[1]+1):
            #bootstrap cluster at the group stability matrix level
            for I in range(optimize_range[0], optimize_range[1]+1):
                stb=L_stability_matrices[L-optimize_range[0],I-optimize_range[0],:,:]
                kmeans=KMeans(n_clusters=M, random_state=0, #set random state to a deterministic initialization
                                    n_init=kmeans_n_init, tol=1e-6).fit(stb)
                labels=kmeans.labels_
                M_labels[I-optimize_range[0],L-optimize_range[0],:]=labels
                scores=stability_scores(stb, labels)
                stability_crit=stability_criterion(scores, labels)
                M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=stability_crit

        overall_stability_evaluation.append(M_stability_evaluation)
        all_labels.append(M_labels)

        #now finding the optimal clustering solution for each M
        optimal_stab=M_stability_evaluation[0,0]
        I_optimal=optimize_range[0]
        L_optimal=optimize_range[0]
        for I in range(optimize_range[0], optimize_range[1]+1):
            for L in range(optimize_range[0], optimize_range[1]+1):
                if M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]>optimal_stab:
                    optimal_stab=M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]
                    I_optimal=I
                    L_optimal=L

        optimal_clustering_solutions[(M-cluster_range[0]),:]=np.array([optimal_stab, I_optimal, L_optimal])

    final_output={'optimal_clustering_solutions':optimal_clustering_solutions, 'overall_stability_evaluation': overall_stability_evaluation, 'optimize_ranges':optimize_ranges,'group_stability_matrices':L_stability_matrices, 'individual_stability_matrices':individual_stability_matrices, 'all_labels':all_labels}
    save_obj(final_output, out_path+'/BASC/'+filename)
    os.remove(out_path+'/BASC/tmp/temp_group_BASC.pkl')
    os.remove(out_path+'/BASC/tmp/temp_individual_BASC.pkl')
    return final_output



'''
This function takes n FC_matrices as inputs, runs multilevel bootstraping and
clustering, over a range of clusters M, as defined in the BASC algorithm.
'''
def old_run_BASC(FC_matrices, cluster_range, num_iter, kmeans_n_init, resampling_prop=0.8, evaluate_over_range=False, verbose=True, out_path='', filename='BASC'):
    np.seterr(all='raise')

    os.makedirs(out_path+'/BASC/tmp', exist_ok=True)

    num_subject=np.size(FC_matrices, 0)
    num_roi=np.size(FC_matrices, 1)
    num_clusters=len(range(cluster_range[0], cluster_range[1]+1))

    optimize_ranges=[] #keep track of the ranges of cluster num over which it was evaluated
    individual_stability_matrices=[] #stack all the stability matrices obtained from bootstrap at the individual level
    group_stability_matrices=[]
    overall_stability_evaluation=[]
    optimal_clustering_solutions=np.zeros([num_clusters, 3])
    all_labels=[]

    #try to recover previous runtime to avoid repeating
    previous_run=False
    try:
        temp_BASC=load_obj(out_path+'/BASC/tmp/temp_BASC')
        previous_run=True
        print('LOADING PREVIOUS RUN.')
    except:
        print('No previous run.')

    if previous_run:
        optimal_clustering_solutions=temp_BASC.get('optimal_clustering_solutions')
        overall_stability_evaluation=temp_BASC.get('overall_stability_evaluation')
        optimize_ranges=temp_BASC.get('optimize_ranges')
        group_stability_matrices=temp_BASC.get('group_stability_matrices')
        individual_stability_matrices=temp_BASC.get('individual_stability_matrices')
        all_labels=temp_BASC.get('all_labels')
        low_bound=temp_BASC.get('current_iteration')

    else:
        low_bound=cluster_range[0]

    #iterate over a range of k
    for M in range(low_bound, cluster_range[1]+1):

        #if the I_DySC and G_DySC k number are optimized, evaluate over a range
        #of between [70%k, 130%k]
        if evaluate_over_range:
            optimize_range=[round(0.7*M), round(1.3*M)]
            if optimize_range[0]<2:
                optimize_range[0]=2
        else:
            optimize_range=[M, M]
        optimize_ranges.append(optimize_range)

        length_optimize_range=len(range(optimize_range[0], optimize_range[1]+1))


        M_individual_stability_matrices=np.empty([num_subject, length_optimize_range,num_roi, num_roi])
        #bootstrap cluster at the individual matrix level
        for sub in range(num_subject):
            error=True
            while(error):
                try:
                    if verbose:
                        print("BASC for subject #"+str(sub))
                    subsample_labels=bootstrap_over_range(FC_matrices[sub,:,:], boot_axis=0, cluster_axis=0, num_iter=num_iter, cluster_range=cluster_range, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="BASC Subject"+str(sub))
                    #derive individual stability matrices
                    for I in range(optimize_range[0], optimize_range[1]+1):
                        stb=stability_matrix(subsample_labels[:,(I-optimize_range[0]),:])
                        M_individual_stability_matrices[sub,(I-optimize_range[0]),:,:]=stb

                    error=False
                except FloatingPointError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
        individual_stability_matrices.append(M_individual_stability_matrices)


        #here will iterate through the clustering solutions for I and L, cluster
        #the group stability matrix with M, and calculate the stability of this solution
        M_stability_matrices=[]
        M_labels=np.zeros([length_optimize_range, length_optimize_range, num_roi])
        M_stability_evaluation=np.zeros([length_optimize_range, length_optimize_range]) #I,L solutions
        for I in range(optimize_range[0], optimize_range[1]+1):
            I_stb=M_individual_stability_matrices[:,I-optimize_range[0],:,:]
            #apply bootstrap cluster over the whole set of states (G_DySC)
            #outputs num_iter*k_range*num_vertices label matrix
            #num_vertices is the number of states, so changes for each I
            error=True
            while(error):
                try:
                    group_subsample_labels=bootstrap_over_range(I_stb, boot_axis=0, cluster_axis=1, num_iter=num_iter, cluster_range=cluster_range, kmeans_n_init=kmeans_n_init, resampling_prop=resampling_prop, verbose=verbose, prefix="BASC Group ")
                    L_stability_matrices=np.zeros([length_optimize_range, num_roi,num_roi])
                    for L in range(optimize_range[0], optimize_range[1]+1):
                        #calculate the stability matrix over the given I and L cluster num,
                        #and cluster over this stability matrix with M clusters
                        stb=stability_matrix(group_subsample_labels.take((L-optimize_range[0]), 1))
                        L_stability_matrices[L-optimize_range[0],:,:]=stb
                        kmeans=KMeans(n_clusters=M, random_state=0, #set random state to a deterministic initialization
                                            n_init=kmeans_n_init, tol=1e-6).fit(stb)
                        labels=kmeans.labels_
                        M_labels[I-optimize_range[0],L-optimize_range[0],:]=labels
                        scores=stability_scores(stb, labels)
                        stability_crit=stability_criterion(scores, labels)
                        M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]=stability_crit
                    error=False
                except ValueError:
                    error=True
                    print('A division by 0 has occurred. Retry the bootstrap.')
            #get the labels obtained over a range, for each I states
            M_stability_matrices.append(L_stability_matrices)
        group_stability_matrices.append(M_stability_matrices)
        overall_stability_evaluation.append(M_stability_evaluation)
        all_labels.append(M_labels)


        #now finding the optimal clustering solution for each M
        optimal_stab=M_stability_evaluation[0,0]
        I_optimal=optimize_range[0]
        L_optimal=optimize_range[0]
        for I in range(optimize_range[0], optimize_range[1]+1):
            for L in range(optimize_range[0], optimize_range[1]+1):
                if M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]>optimal_stab:
                    optimal_stab=M_stability_evaluation[I-optimize_range[0],L-optimize_range[0]]
                    I_optimal=I
                    L_optimal=L

        optimal_clustering_solutions[(M-low_bound),:]=np.array([optimal_stab, I_optimal, L_optimal])

        temp_BASC={'optimal_clustering_solutions':optimal_clustering_solutions, 'overall_stability_evaluation': overall_stability_evaluation, 'optimize_ranges':optimize_ranges,'group_stability_matrices':group_stability_matrices, 'individual_stability_matrices':individual_stability_matrices, 'all_labels':all_labels, 'current_iteration':M}
        save_obj(temp_BASC, out_path+'/BASC/tmp/temp_BASC')


    final_output={'optimal_clustering_solutions':optimal_clustering_solutions, 'overall_stability_evaluation': overall_stability_evaluation, 'optimize_ranges':optimize_ranges,'group_stability_matrices':group_stability_matrices, 'individual_stability_matrices':individual_stability_matrices, 'all_labels':all_labels}
    save_obj(final_output, out_path+'/BASC/'+filename)
    os.remove(out_path+'/BASC/tmp/temp_BASC.pkl')
    return final_output


'''
Computes bootstrap analysis of stable clusters (BASC) at the group level as defined in
Bellec et al. 2010. It takes as input stability matrices from individual
subjects. Then, through bootstrapping, it iteratively
randomly take out a defined portion of the sample n times to obtain n subsamples,
from which the average stability matrix is then computed. Then, a clustering
algorithm, such as hierarchical agglomerative clustering, is applied onto the
subsample average stability matrices to generate new partitions from each
subsample. Finally, the stability matrix is calculated from these n partitions,
and clustering can be applied again to generate a final clustering solution.
Here, instead of using hierarchical clustering, k-means will be used with a
defined number of centroids.
'''



'''
Bootstrap over a range of k solution for the clustering of the resamples. Gives
one stability matrix per k in range, and their labels. Then the output labels
'''
def multiscale_bootstrap(matrix, boot_axis, cluster_axis, num_iter, cluster_range, kmeans_n_init, resampling_prop=0.8, verbose=True, prefix=""):
    if num_iter<100:
        print('WITH A SMALL NUMBER OF BOOTSTRAPS, THERE IS A HIGH CHANCE OF HAVING A DIVISION BY 0 IN THE STABILITY MATRIX.')

    num_vertices=int(np.size(matrix, cluster_axis))
    num_clusters=len(range(cluster_range[0], cluster_range[1]))+1
    error=True
    while(error):
        try:
            all_group_subsample_labels=np.zeros([num_clusters, num_clusters, num_vertices])
            individual_subsample_labels=bootstrap_over_range(matrix, boot_axis, cluster_axis, num_iter, cluster_range, kmeans_n_init, resampling_prop, verbose, prefix=prefix)
            for I in range(cluster_range[0], cluster_range[1]+1):
                stb=stability_matrix(individual_subsample_labels[:,I-cluster_range[0],:])
                group_subsample_labels=np.zeros([num_clusters, np.size(stb, 0)])
                for L in range(cluster_range[0], cluster_range[1]+1):
                    kmeans=KMeans(n_clusters=L, random_state=0, #set random state to a deterministic initialization
                                        n_init=kmeans_n_init, tol=1e-6).fit(stb)
                    group_subsample_labels[L-cluster_range[0],:]=kmeans.labels_
                all_group_subsample_labels[I-cluster_range[0],:,:]=group_subsample_labels

            error=False
        except:
            error=True
            print('A division by 0 has occurred. Retry the bootstrap.')
    return all_group_subsample_labels, individual_subsample_labels
'''
        scores=stability_scores(stb, labels)
        stability_evaluation[c]=stability_criterion(scores, labels)
        stability_matrices.append(stb)
        c+=1
    optimal_k=range(cluster_range[0], cluster_range[1]+1)[np.argmax(stability_evaluation)]
    optimal_label=all_labels[np.argmax(stability_evaluation), :]
    optimal_stb=stability_matrices[np.argmax(stability_evaluation)]
    return optimal_label, optimal_k, stability_evaluation, optimal_stb
'''


'''
Bootstrap over a cluster range, and output the labels for each k.
Take an input matrix, bootstrap over a given axis, cluster
the subsamples over a given axis and outputs the set of partitions. If the
axis for boot and cluster are the same, will add None in the partition arrays
for the indices which were ommited in the subsample.
'''
def bootstrap_over_range(matrix, boot_axis, cluster_axis, num_iter, cluster_range, kmeans_n_init, resampling_prop=0.8, verbose=True, prefix=""):
    import random
    boot_range=int(np.size(matrix, boot_axis))
    if cluster_axis==boot_axis:
        num_vertices=int(np.size(matrix, boot_axis)*resampling_prop)
    else:
        num_vertices=int(np.size(matrix, cluster_axis))
    all_subsample_labels=np.zeros([num_iter, len(range(cluster_range[0], cluster_range[1]+1)), np.size(matrix, cluster_axis)])
    for i in range(0,num_iter):
        #generates a random selection of resampling_prop % of the sample
        random_order=random.sample(range(boot_range), boot_range)
        subsample_matrices=matrix.take(random_order[:int(boot_range*resampling_prop)], axis=boot_axis)

        shape=subsample_matrices.take(0, cluster_axis).shape
        if len(shape)>1:
            #reshape the 3D matrix into a set of vectors, one per vertice to cluster
            subsample=np.zeros([num_vertices, shape[0]*shape[1]])
            for sub in range(num_vertices):
                subsample[sub,:]=np.resize(subsample_matrices.take(sub, cluster_axis), shape[0]*shape[1])
        else:
            subsample=subsample_matrices


        subsample_labels=np.empty([len(range(cluster_range[0], cluster_range[1]+1)), np.size(matrix, cluster_axis)])
        for k in range(cluster_range[0], cluster_range[1]+1):
            kmeans=KMeans(n_clusters=k, random_state=0, #set random state to a deterministic initialization
                                n_init=kmeans_n_init, tol=1e-6).fit(subsample)
            labels=kmeans.labels_

            if cluster_axis==boot_axis:
                #reorder the random_order
                sort_order=np.argsort(random_order[:int(boot_range*resampling_prop)]) #obtain the indices for a sorted array
                sorted_order=np.sort(random_order[:int(boot_range*resampling_prop)])
                reordered_labels=fc_utils.reorder(labels, sort_order)
                c=0
                while(sorted_order[0]!=c):
                    subsample_labels[(k-cluster_range[0]),c]=-1
                    c+=1
                for j in range(0, len(sorted_order)):
                    while(sorted_order[j]!=c):
                        subsample_labels[(k-cluster_range[0]),c]=-1
                        c+=1
                    subsample_labels[(k-cluster_range[0]),c]=int(reordered_labels[j])
                    c+=1

                while(c<boot_range):
                    subsample_labels[(k-cluster_range[0]),c]=-1
                    c+=1
            else:
                subsample_labels[(k-cluster_range[0]),:]=labels
            if verbose:
                print(prefix+"Eval for "+str(k)+" clusters: boostrap iter #"+str(i+1))
        all_subsample_labels[i,:,:]=subsample_labels
    return all_subsample_labels



'''
Bootstrap function. Take an input matrix, bootstrap over a given axis, cluster
the subsamples over a given axis and outputs the set of partitions. If the
axis for boot and cluster are the same, will add None in the partition arrays
for the indices which were ommited in the subsample.
'''

def bootstrap_cluster(matrix, boot_axis, cluster_axis, num_iter, kmeans_k, kmeans_n_init, resampling_prop=0.8, verbose=True, prefix=""):
    import random
    boot_range=int(np.size(matrix, boot_axis))
    if cluster_axis==boot_axis:
        num_vertices=int(np.size(matrix, boot_axis)*resampling_prop)
    else:
        num_vertices=int(np.size(matrix, cluster_axis))
    subsample_labels=np.empty([num_iter, np.size(matrix, cluster_axis)])
    for i in range(0,num_iter):
        #generates a random selection of resampling_prop % of the sample
        random_order=random.sample(range(boot_range), boot_range)
        subsample_matrices=matrix.take(random_order[:int(boot_range*resampling_prop)], axis=boot_axis)

        #reshape the 3D matrix into a set of vectors, one per vertice to cluster
        shape=subsample_matrices.take(0, cluster_axis).shape
        if len(shape)>1:
            #reshape the 3D matrix into a set of vectors, one per vertice to cluster
            subsample=np.zeros([num_vertices, shape[0]*shape[1]])
            for sub in range(num_vertices):
                subsample[sub,:]=np.resize(subsample_matrices.take(sub, cluster_axis), shape[0]*shape[1])
        else:
            subsample=subsample_matrices

        kmeans=KMeans(n_clusters=kmeans_k, random_state=0, #set random state to a deterministic initialization
                            n_init=kmeans_n_init, tol=1e-6).fit(subsample)
        labels=kmeans.labels_
        if cluster_axis==boot_axis:
            #reorder the random_order
            sort_order=np.argsort(random_order[:int(boot_range*resampling_prop)]) #obtain the indices for a sorted array
            sorted_order=np.sort(random_order[:int(boot_range*resampling_prop)])
            reordered_labels=fc_utils.reorder(labels, sort_order)
            c=0
            while(sorted_order[0]!=c):
                subsample_labels[i,c]=-1
                c+=1
            for j in range(0, len(sorted_order)):
                while(sorted_order[j]!=c):
                    subsample_labels[i,c]=-1
                    c+=1
                subsample_labels[i,c]=int(reordered_labels[j])
                c+=1

            while(c<boot_range):
                subsample_labels[i,c]=-1
                c+=1

        else:
            subsample_labels[i,:]=labels
        if verbose:
            print(prefix+"Eval for "+str(kmeans_k)+" clusters: boostrap iter #"+str(i+1))
    return subsample_labels

'''
Calculates the stability criterion as defined in Bellec et al. 2010. This
corresponds to the average of the stability of scores of ROIs within their
designeted cluster, relative to the maximal stability score obtain in other
clusters.
'''
def stability_criterion(scores, group_labels):
    N=np.size(scores, 0)
    stability=np.empty([N])
    for i in range(N):
        label_score=scores[i,group_labels[i]]
        out_cluster_max=np.max(np.delete(scores[i,:], group_labels[i]))
        stability[i]=label_score-out_cluster_max
    return np.mean(stability)



'''
Evaluate the stability scores of each ROI in the group stability matrix,
relative to each clusters. The stability score is defined as the average
stability of a given ROI with all other ROIs belonging to a cluster. This
function calculates the stability scores of each ROI for each clusters.
'''
def stability_scores(group_stability_matrix, group_labels):
    C=np.max(group_labels)+1
    N=np.size(group_stability_matrix, 0)
    scores=np.empty([N, C])
    for c in range(C):
        c_indices=c==group_labels
        c_num_labels=np.sum(c_indices)
        if c_num_labels>1:
            for i in range(N):
                cum_prob=0
                for cur_c in range(N):
                    if c_indices[cur_c] and cur_c!=i:
                        cum_prob+=group_stability_matrix[i,cur_c]
                scores[i,c]=cum_prob/(c_num_labels-1)
        else:
            #here I assume that there shouldn't be clusters of only one ROI
            #hence, I'll give no weight to these solutions
            for i in range(N):
                scores[i,c]=0
    return scores



'''
Stability matrix is the probability that a pair of two regions belong to the
same cluster. The stability matrix is derived from taking the average from
the adjacency matrix from in individual partitions, and dividing by the
number of partitions p.
'''

def stability_matrix(partitions):
    #takes as input a p_partitions*n_nodes matrix
    if len(partitions.shape)==2:
        stability_matrix=np.zeros([np.size(partitions, 1), np.size(partitions, 1)])
        num_boot=np.zeros([np.size(partitions, 1), np.size(partitions, 1)])
        for p in range(np.size(partitions, 0)):
            adj=adjacency_matrix(partitions[p,:])
            for i in range(np.size(adj, 0)):
                for j in range(np.size(adj, 1)):
                    if adj[i,j]!=-1:
                        stability_matrix[i,j]+=adj[i,j]
                        num_boot[i,j]+=1
        stability_matrix=stability_matrix/num_boot
        return stability_matrix
    else:
        # in the case where there is only a single partition
        return adjacency_matrix(partitions)


'''
Computes the adjacency matrix of a vector. Returns 1 when two
indices have the same value, and 0 if not.
'''

def adjacency_matrix(vector):
    matrix=np.zeros([len(vector), len(vector)])
    for i in range(len(vector)):
        for j in range(len(vector)):
            if vector[i]==-1 or vector[j]==-1:
                matrix[i,j]=-1
            else:
                matrix[i,j]=vector[i]==vector[j]
    np.fill_diagonal(matrix, 0)

    return matrix
