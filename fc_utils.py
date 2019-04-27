import numpy as np

def cumul(n):
    if n == 0:
        return 0
    else:
        return n + cumul(n-1)

def compute_zscores(array):
    #takes 2D array t*n, and returns n*n z-transformed correlation matrix
    r=np.corrcoef(np.transpose(array))
    np.fill_diagonal(r, 0.5)
    z=0.5*np.log((1+r)/(1-r))
    np.fill_diagonal(z, 0)
    return z

#unwarp an FC matrix to a vector; deletes upper half of the diagonal
def unwarp_to_vector(matrix):
    vector=np.empty([int((len(matrix)**2-len(matrix))/2)])
    vector[0:1]=matrix[1,:1]
    for i in range(1,len(matrix)):
        vector[cumul(i-1):cumul(i)]=matrix[i,:i]
    return vector


def get_FC_vector(timeseries, metric='r'):
    if metric=='r':
        correlation_matrix=np.corrcoef(np.transpose(timeseries))
    #get FC as a vector of z scores from timeseries (t*n matrix)
    elif metric=='z':
        correlation_matrix=compute_zscores(timeseries)
    else:
        raise ValueError('Metric specification must be r or z.')
    np.fill_diagonal(correlation_matrix, None)
    FC_vector=unwarp_to_vector(correlation_matrix)
    return FC_vector

def stack_matrices(FC_vectors, num_roi):
    num_sample=np.size(FC_vectors, 0)
    stacked_matrices=np.zeros([num_roi, num_roi*num_sample])
    for i in range(0,num_sample):
        matrix=recover_matrix(FC_vectors[i,:], num_roi)
        stacked_matrices[:,(i)*num_roi:(i+1)*num_roi]=matrix
    return stacked_matrices

def recover_matrix(FC_vector,num_roi):
    matrix=np.zeros([num_roi, num_roi])
    for i in range(1,num_roi):
        matrix[i,:i]=FC_vector[cumul(i-1):cumul(i)]
        matrix[:i,i]=FC_vector[cumul(i-1):cumul(i)]
    np.fill_diagonal(matrix, 0)
    return matrix

def reorder(orig, sort_order):
    #reorder indices of a vector or matrix
    if len(orig.shape)>1:
        reordered_FC=np.empty([np.size(orig,0), np.size(orig,1)])
        for i in range(0, np.size(orig, 0)):
            for j in range(0, np.size(orig, 1)):
                reordered_FC[i,j]=orig[sort_order[i],sort_order[j]]
    else:
        reordered_FC=np.empty([np.size(orig,0)])
        for i in range(0, np.size(orig, 0)):
            reordered_FC[i]=orig[sort_order[i]]
    return reordered_FC

def within_cluster_var(stacked_matrices, labels, num_centroids):
    #compute within cluster variance
    cluster_var=np.zeros([num_centroids, np.size(stacked_matrices, 1)])
    for i in range(0,num_centroids):
        cluster_tmp=[];
        for j in range(0,len(labels)):
            if labels[j]==i:
                cluster_tmp.append(stacked_matrices[j,:])
        cluster_array=np.zeros([len(cluster_tmp), np.size(stacked_matrices, 1)])
        for j in range(0,len(cluster_tmp)):
            cluster_array[j,:]=cluster_tmp[j]
        cluster_var[i,:]=np.var(cluster_array, 0)
    mean_var=np.mean(cluster_var, (0,1));
    return [cluster_var, mean_var]

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

'''
def sliding_window_dFC(subject_timeseries, window_length, num_nodes):
    #extracts the FC sliding windows from a subject, with windows spaced by
    #1 TR
    num_windows=np.size(subject_timeseries, 0)-(window_length-1)
    vector_length=cumul(num_nodes-1)
    subject_windows=np.empty([num_windows, vector_length])
    for i in range(0, num_windows):
        window_array=subject_timeseries[i:(i+window_length-1), :]
        vector_r=get_FC_vector(window_array)
        subject_windows[i,:]=vector_r
    return subject_windows
'''

def sliding_window_dFC(subject_timeseries, window_length, window_step, num_nodes):
    #extracts the FC sliding windows from a subject, with windows spaced by
    #window_step
    num_TRs=np.size(subject_timeseries, 0)
    num_windows=int(num_TRs/window_step)
    while(window_length+(num_windows-1)*window_step>num_TRs):
        num_windows-=1
    vector_length=cumul(num_nodes-1)
    subject_windows=np.empty([num_windows, vector_length])
    for i in range(0, num_windows):
        window_array=subject_timeseries[i*window_step:((i*window_step)+window_length), :]
        vector_r=get_FC_vector(window_array)
        subject_windows[i,:]=vector_r
    return subject_windows


def filter_windows(vectored_windows,lowcut, highcut):
    filtered=np.zeros(vectored_windows.shape)
    for sub in range(np.size(vectored_windows,0)):
        print(sub)
        for edge in range(np.size(vectored_windows,2)):
            data=vectored_windows[sub,:,edge]
            filtered[sub,:,edge]=butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=1)
    return filtered

def window_var(windows):
    var=np.zeros([np.size(windows, 0), np.size(windows, 2)])
    for i in range(np.size(windows, 0)):
        var[i,:]=np.std(windows[i,:,:], 0)
    return var

def roi_pair_temporal_covariance(vectored_windows, num_roi,roi1,roi2):
    #between ROI pairs, will evaluate the correlation in their connectivity changes across all other ROIs
    #the length of one ROI pair values is (num_nodes-2)*num_windows
    num_windows=np.size(vectored_windows,0)
    covar=np.zeros([2,(num_roi-2)*num_windows])
    c=0
    for i in range(num_windows):
        window=recover_matrix(vectored_windows[i,:],num_roi)
        index1=np.arange(num_roi)!=roi1
        index2=np.arange(num_roi)!=roi2
        correct_indices=index1*index2
        for index in range(len(correct_indices)):
            if correct_indices[index]:
                covar[0,c]=window[roi1,index]
                covar[1,c]=window[roi2,index]
                c+=1
    from scipy import stats
    return stats.pearsonr(covar[0,:],covar[1,:])[0]


def connectivity_covariance_correlations(vectored_windows, num_roi):
    correlations=np.empty([int((num_roi**2-num_roi)/2)])
    c=0
    for roi1 in range(num_roi):
        for roi2 in range(roi1):
            correlations[c]=roi_pair_temporal_covariance(vectored_windows, num_roi,roi1,roi2)
            c+=1
    return correlations



def visualize_FC(ax, FC_vector, num_roi, reorder_roi=False, labels=None, title=None, vmax=None, vmin=None):
#    from nilearn import plotting
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if reorder_roi:
        sort_order=np.argsort(labels) #obtain the indices for a sorted array
        sorted_labels=np.sort(labels)
        matrix=reorder(recover_matrix(FC_vector,num_roi), sort_order)
        im = ax.imshow(matrix, cmap='coolwarm',vmax=vmax, vmin=vmin)
        if title!=None:
            ax.set_title(title)

#        ax.set_yticks(np.arange(len(sorted_labels)), sorted_labels)
        # Create divider for existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax3, with 20% width of ax3
        cax = divider.append_axes("right", size="7%", pad=0.05)
        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        cbar = plt.colorbar(im, cax=cax) #ticks=MultipleLocator(0.2), format="%.2f")
        # Remove xticks from ax3
        ax.xaxis.set_visible(False)
#        cbar=ax.colorbar(spacing='proportional')#,ticks=ticks)
        cbar.set_label('Z Scores')
    else:
        matrix=recover_matrix(FC_vector,num_roi)
        plt.imshow(matrix, cmap='coolwarm',vmax=vmax, vmin=vmin)
        cbar=plt.colorbar(spacing='proportional')#,ticks=ticks)
        cbar.set_label('Z Scores')

def seaborn_visualize(ax, FC_vector, num_roi, reorder_roi=False, cmap='coolwarm', center=True, labels=None, title=None, vmax=None, vmin=None):
#    from nilearn import plotting
    import seaborn as sns
    sns.set(font_scale=1)

    if reorder_roi=='hierarchical':
        #attribute colors
        matrix=recover_matrix(FC_vector,num_roi)
        if center:
            g=sns.clustermap(matrix, cmap=cmap,vmax=vmax, vmin=vmin, center=0,figsize=(6, 6),
                        linewidths=0, xticklabels=False, yticklabels=False, square=False,
                        cbar_kws={"ticks":[vmin,0,vmax], "shrink": 1, 'label': 'Z Scores'},
                        # Turn ON the clustering
                        row_cluster=True, col_cluster=True)
            g.ax_col_dendrogram.set_visible(False)
        else:
            g=sns.clustermap(matrix, cmap=cmap,vmax=vmax, vmin=vmin, figsize=(6, 6),
                        linewidths=0, xticklabels=False, yticklabels=False, square=False,
                        cbar_kws={"ticks":[vmin,0,vmax], "shrink": 1, 'label': 'Z Scores'},
                        # Turn ON the clustering
                        row_cluster=True, col_cluster=True)
            g.ax_col_dendrogram.set_visible(False)
        g.fig.suptitle(title)

        return g

    elif reorder_roi:
        sort_order=np.argsort(labels) #obtain the indices for a sorted array
        sorted_labels=np.sort(labels)
        plot_labels=list(sorted_labels.astype(int))
        colors = ['red','green','blue','cyan', 'magenta', 'yellow', 'purple', 'black', 'purple', 'pink', 'grey', 'orange']

        network_colors=[]
        for i in range(len(plot_labels)):
            network_colors.append(colors[plot_labels[i]])

        #attribute colors
        matrix=reorder(recover_matrix(FC_vector,num_roi), sort_order)
        if center:
            g=sns.clustermap(matrix, cmap=cmap,vmax=vmax, vmin=vmin, center=0,figsize=(6, 6),
                        linewidths=0, xticklabels=False, yticklabels=False, square=False,
                        cbar_kws={"ticks":[vmin,0,vmax], "shrink": 1, 'label': 'Z Scores'},
                        # Turn off the clustering
                        row_cluster=False, col_cluster=False,
                        # Add colored class labels
                        row_colors=network_colors,col_colors=network_colors)
        else:
            g=sns.clustermap(matrix, cmap=cmap,vmax=vmax, vmin=vmin, figsize=(6, 6),
                        linewidths=0, xticklabels=False, yticklabels=False, square=False,
                        cbar_kws={"ticks":[vmin,0,vmax], "shrink": 1, 'label': 'Z Scores'},
                        # Turn off the clustering
                        row_cluster=False, col_cluster=False,
                        # Add colored class labels
                        row_colors=network_colors,col_colors=network_colors)
        g.fig.suptitle(title)

        return g


    else:
        matrix=recover_matrix(FC_vector,num_roi)
        im = sns.heatmap(matrix, ax=ax, cmap='coolwarm',vmax=vmax, vmin=vmin, center=0,
                    square=True, linewidths=0, xticklabels=False, yticklabels=False, cbar_kws={"shrink": .5, 'label': 'Z Scores'}).set_title(title)


def edgewise_ttest(groupA, groupB):
    #edge-wise t-tests between each groups, corrected for false discovery rate
    from scipy import stats
    from statsmodels.stats.multitest import fdrcorrection
    t_stats=np.zeros(np.size(groupA, 1))
    for i in range(0,np.size(groupA, 1)):
        groupA_values=groupA[:,i]
        groupB_values=groupB[:,i]
        t2, p2 = stats.ttest_ind(groupA_values, groupB_values,
                                 equal_var=True,nan_policy='raise') #assume equal variance in subjects
        t_stats[i]=p2

    [rejected, fdr_corrected]=fdrcorrection(t_stats, alpha=0.05, method='indep', is_sorted=False) #alpha=FDR threshold (5% for 0.05)
    return fdr_corrected, t_stats

#get expression of states through linear regression
def states_temporal_regression(vectored_windows, vectored_states, num_nodes, verbose=True):
    num_states=np.size(vectored_states, 0)
    num_windows=np.size(vectored_windows, 0)

    #linear regression for each window for each state
    states_timecourses=np.zeros([num_states,num_windows])
    from sklearn import linear_model

    for i in range(num_states):
        state=vectored_states[i,:]
        for j in range(num_windows):
            window=vectored_windows[j,:]
            # Create linear regression object
            regr = linear_model.LinearRegression()
            # Fit the parameter
            regr.fit(window.reshape(-1, 1), state.reshape(-1, 1))
            states_timecourses[i,j]=regr.coef_
        if verbose:
            print('State '+str(i)+' computed')
    return states_timecourses


def evaluate_PCA_stats(vectored_windows, num_iter, verbose=True):
    #evaluate t-stats p-values relative to null model obtained from permutations
    #and fdr correct the p-values
    import scipy.stats
    from statsmodels.stats.multitest import fdrcorrection
    [components, data_explained_variance, transformed]=windows_PCA(vectored_windows)
    null_dist=build_null_PCA(vectored_windows, num_iter, verbose=verbose)
    dist_mean=np.mean(null_dist, 0)
    t, p=scipy.stats.ttest_1samp(data_explained_variance, dist_mean)
    [rejected, fdr_corrected]=fdrcorrection(p, alpha=0.05, method='indep', is_sorted=False) #alpha=FDR threshold (5% for 0.05)
    return components, data_explained_variance, transformed, fdr_corrected



'''
PCA on windows
'''
def windows_PCA(vectored_windows):
    #takes as input a stack of vectored windows with n_windows x n_edges
    from sklearn.decomposition import PCA
    pca=PCA()
    transformed=pca.fit_transform(vectored_windows)
    components=pca.components_
    explained_variance=pca.explained_variance_ratio_
    return components, explained_variance, transformed

def build_null_PCA(matrix, num_iter, verbose=True):
    #takes as input n_samples x n_features matrix, and permutes the features
    if np.size(matrix, 0)<np.size(matrix, 1):
        n_comp=np.size(matrix, 0)
    else:
        n_comp=np.size(matrix, 1)
    null_distribution=np.zeros([num_iter, n_comp])
    for i in range(num_iter):
        [components, explained_variance, transformed]=windows_PCA(permutation(matrix))
        null_distribution[i,:]=explained_variance
        if verbose:
            print("Null model iteration "+str(i))
    return null_distribution

def permutation(matrix):
    #takes as input n_samples x n_features matrix, and permutes the features
    import random
    n_samples=np.size(matrix, 0)
    n_features=np.size(matrix, 1)
    permuted=np.zeros([n_samples,n_features])
    for i in range(n_samples):
        random_order=random.sample(range(n_features), n_features)
        permuted[i,:]=matrix[i,random_order]
    return permuted
