# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:10:57 2020

@author: ISP
"""
#%%
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.mixture import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.testing import ignore_warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import decomposition
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.mixture import *
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import *
from sklearn.cluster import *
from sklearn.mixture import *
from sklearn import cluster 


#%% PCA 관련 함수
# std_concat, deviation = X, deviation 
def PCA_opt(std_concat, deviation):
    pca = decomposition.PCA(n_components=deviation) #0.95: deviation
    std_pca = pca.fit_transform(std_concat)
    pca_result = pd.DataFrame(std_pca)
    pca_information =  pca.explained_variance_ratio_
    pca_covar = pca.get_covariance()
    print('eigen_value :', pca.explained_variance_)
    print('explained variance ratio :',pca_information) # the quantiy of information included
    #reconstruction of latent
    reconstructed_pca = pca.inverse_transform(pca_result)
    #proper choice of dimension
    cumsum = np.cumsum(pca_information)
    d = np.argmax(cumsum >= deviation) + 1
    print('choice of # of dimension:', d)
    #PCA loss
    return(pca_result, pca, pca_covar) 

#%% 클러스터링 관련 함수
def Clustering_models(n_clusters):
    two_means = MiniBatchKMeans(n_clusters=n_clusters)
    dbscan = DBSCAN(eps=0.15)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
    ward = AgglomerativeClustering(n_clusters=n_clusters)
    affinity_propagation = AffinityPropagation(damping=0.9, preference=-200)
    gmm = GaussianMixture(n_components=n_clusters, init_params='random', random_state=0, tol=1e-9)
    
    clustering_algorithms = (
        ('K-Means', two_means),
        # ('DBSCAN', dbscan),
        ('Hierarchical Clustering', ward),
        #('Affinity Propagation', affinity_propagation),
        ('Spectral Clustering', spectral),
        ('Gaussian Mixture', gmm),
    )
    return(clustering_algorithms)


# # 4. 덴드로그램 시각화 : 군집수 결정
# import matplotlib.pyplot as plt
# plt.figure( figsize = (25, 10) )
# dendrogram(clusters, leaf_rotation=90, leaf_font_size=12,)
# # leaf_rotation=90 : 글자 각도
# # leaf_font_size=20 : 글자 사이즈
# plt.show() 

# 최적 클러스터 실행
def exectue_cluster(clustering_algorithms, n_clusters, PCA_data, label):
    plot_num = 1
    
    clster_pred = []
    "Plotting"
    for j, (name, algorithm) in enumerate(clustering_algorithms):
        with ignore_warnings(category=UserWarning):
            algorithm.fit(X)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)
        # plt.subplot(len(clustering_algorithms), 1, plot_num)
        
        plt.figure()
        plt.title(name)
        # centers = algorithm.cluster_centers_
        colors = plt.cm.tab10(np.arange(20, dtype=int))
        "PCA dimension reduction needed"
        plt.scatter(PCA_data[:, 0], PCA_data[:, 1], s=5, color=colors[y_pred])
        # plt.scatter(x=centers[:,0], y=centers[:,1], marker='D', c='r')
        
        # plt.scatter(PCA_data[idx, 0], PCA_data[idx, 1], label = i, s=5)

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        # plt.xticks(())
        # plt.yticks(())
        plot_num += 1
        # plt.legend()
        plt.tight_layout()
        plt.show()
        
        clster_pred.append(y_pred)
    return(clster_pred)
        
# 실루엣 평가
def plot_silhouette(X, test_no, PCA_data):
    # test_no = 8
    
    colors = plt.cm.tab10(np.arange(40, dtype=int))
    plt.figure(figsize=(6, 8))
    list_silhouette = []
    list_clusters = []
    for i in range(test_no):
        n = i+2
        two_means = MiniBatchKMeans(n_clusters=n) #KMeans
        # dbscan = DBSCAN(eps=0.15)
        spectral = SpectralClustering(n_clusters=n, affinity="nearest_neighbors")
        ward = AgglomerativeClustering(n_clusters=n)
        # affinity_propagation = AffinityPropagation(damping=0.9, preference=-200)
        gmm = GaussianMixture(n_components=n, init_params='random', random_state=0, tol=1e-9)
        
        "Select_model"
        model = two_means 
        
        
        cluster_labels = model.fit_predict(X)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        silhouette_avg = sample_silhouette_values.mean()
        list_clusters.append(n)
        list_silhouette.append(silhouette_avg)

    # list_silhouette = np.array(list_silhouette)
    plt.plot(np.array(list_clusters),np.array(list_silhouette), "+-")
    plt.title("Sillouete score")
    plt.show()
    
    return(pd.DataFrame(list_clusters), pd.DataFrame(list_silhouette))
#%%
# dunn_index 관련 함수
    
DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def inter_cluster_distances(labels, distances, method='nearest'):

    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
   
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest',
         cdist_method='nearest'):

    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn_value(k_list):
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di
    
    
def  big_s(x, center):
    len_x = len(x)
    total = 0
        
    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)    
    
    return total/len_x

def davisbouldin(k_list, k_centers):

    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0    

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di
    
def dunn_index_test(X, n_test):
    df = pd.DataFrame(X)  
    # K-Means 
    dunn_list = []
    cluster_no_list = []
    for i in range(n_test):
        print("testtest:", i)
        crt_cluster = i+2
        k_means = cluster.KMeans(n_clusters=crt_cluster) 
        k_means.fit(df) #K-means training 
        y_pred = k_means.predict(df) 

        dunn_list.append(dunn_fast(X, y_pred))
        cluster_no_list.append(crt_cluster)
    return(dunn_list, cluster_no_list)

#%% 클러스터 내 용도지역 분포 구하는 데 필요한 함수

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def euclidean_distance(a0, a1):   
    return np.sqrt(np.sum((a0 - a1) ** 2))

def histo(data): ##occurence
    val, cnt = np.unique(data, return_counts=True)
    histo_data_x = np.column_stack((val, cnt))
    histo_data_x = pd.DataFrame(histo_data_x)
    histo_data_x.columns = ['val', 'cnt']
    return(histo_data_x)  

def Probability_X(X):
    Pr_X = X/np.sum(X)
    return(Pr_X)

#%% 특정 클러스터 내 추천 및 plot
def Recommend_test_data(PCA_data, test_data, y, n_clusters, model):

    df = pd.DataFrame(columns=('x','y'))
    df['x'] = PCA_data[:,0]
    df['y'] = PCA_data[:,1]
    
    df_test = df.copy()
    test_no = len(df)
    df_test.loc[test_no] = test_data#test data
    
    #plot
    # sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":200})
    # plt.scatter(df['x'],df['y'], s=200)
    # plt.xlabel("x")
    # plt.ylabel("y")
    
    #
    # X_test = df_test.values
    # model = KMeans(n_clusters=n_clusters).fit(df.values)
    model.fit(df.values)
    # kmeans_test = KMeans(n_clusters=3).fit(X_test)
    df['cluster'] = model.fit_predict(X)
    
    # df['cluster'] = model.labels_
    plt.figure()
    sns.lmplot('x','y', data=df, fit_reg=False, scatter_kws={"s":100}, hue="cluster")
    plt.scatter(df_test['x'][test_no],df_test['y'][test_no], color='red', s=400, marker='*') #test plot
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # plt.legend()
    plt.show()
    
    #거리구하기
    dist_list = pd.DataFrame([0])
    for i in range(df.shape[0]):
        #X
        dist_list.loc[i] = euclidean_distance(test_data, df.values[i,0:2])
    
    #가장 가까운 클러스터 찾기
    near_value, near_idx = find_nearest(dist_list.values, 0.0)
    final_cluster = df['cluster'][near_idx]
    
    #특정 클러스터 내 용도지역 분포 구하기
    df['yongdo'] = y
    histo_yongdo = df[['cluster', 'yongdo']]
    result_classification = []
    
    for i in range(n_clusters):
        current_cluster_yongdo = histo_yongdo[histo_yongdo['cluster'] == i]
        
        histo_data = histo(current_cluster_yongdo['yongdo'])
        Pr_tot = Probability_X(histo_data.cnt)*100
        
        crt_rst_classifi = histo_data.copy()
        crt_rst_classifi['Pr'] = Pr_tot
        result_classification.append(crt_rst_classifi)
    
    # 최종 우선순위 구하기
    result_priority = result_classification[final_cluster]
    result_priority = result_priority.sort_values(by=['Pr'], axis=0, ascending=False)
    result_priority = result_priority.reset_index(drop=True) #Result value
    
    return(result_priority)

#%%  Input: data 지정 => bus / building
"""
concat_aggregation_new_data_total : [0.85216374 0.06644724]
aggregation_new_data_중구_bus병합 : [0.59714867 0.17728957]

new_data_중구_bus병합 : [0.88235818 0.07553704]
concat_new_data_total : [0.91938787 0.03779958]
"""

path = 'D:/33. 대구 빅데이터/2. 분석/Analysis_final_201003/'
file_name = 'concat_aggregation_new_data_total.csv'

data = pd.read_csv(path + file_name, 
                   dtype={'시군구코드': str, '법정동코드': str, '번': str, '지': str}, 
                   encoding='euc-kr')

yongdo_v, yongdo_c= np.unique(data['jijiguCdNm'].values, return_counts = True)

data.rename(columns={"승하차수":"usage_bus"}, inplace=True)
data.rename(columns={"정류소여부":"busstop_exist"}, inplace=True)
# del data['busstop_exist']

data_col_name = data.columns
# col_x = np.concatenate([np.arange(8,21)]) #total
# col_x = np.concatenate([np.arange(8,13), np.arange(16,18),np.arange(20,21)]) #priority 1~4
col_x = np.concatenate([np.arange(8,11),np.arange(16,18),np.arange(20,21)]) #priority 1~3
# col_x = np.concatenate([np.arange(8,11),np.arange(16,18)]) #priority 1~2
feature_name = data_col_name[col_x]

#test_data
# label1 = data['jijiguCdNm'].copy()# make label
# test_data_save = data[label1 == "문화재보존영향검토대상지역"]
# test_data_save = test_data_save.values[0]
#%%
data = data[data_col_name[col_x]]

data.dropna(inplace = True, axis='rows')
data = data.reset_index(drop=True)

label = data['jijiguCdNm'].copy()# make label
data.drop(columns = ['jijiguCdNm'], inplace = True)

test_data = data[label == "도시지역"]
test_data_idx = test_data.index[0]

n_clusters = 6

#%%
""" Preprocessing """
X = data
y = label

distinct_idx = y.str.find("지역").values
X = X.values
y = y.values

# for i in range(0,len(distinct_idx)-1):
#     if distinct_idx[i+1] < 0:
#         y[i+1] = y[i]
#%%
""" Clustering """
X = RobustScaler().fit_transform(X)
# X2 = StandardScaler().fit_transform(X)
# nn=0
# plt.figure()
# plt.plot(X1[:,nn])
# plt.plot(X2[:,nn])
# plt.show()

#%% PCA: dimension reduction
deviation = 2
PCA_data, PCA_model, PCA_covariance = PCA_opt(X, deviation)
test_data = PCA_data.loc[test_data_idx]
PCA_data = PCA_data.values

#%% chk_plot
# plt.figure()
# plt.scatter(X[:, 0], X[:, 2], s=2)
# plt.show()

# plt.figure()
# plt.scatter(PCA_data[:, 0], PCA_data[:, 1], s=2)
# plt.show()

#%% chk_plot: raw label
label, label_count= np.unique(y, return_counts = True)
plt.figure()
for i, v in enumerate(label):
# for i, v in enumerate(label[10:14]):
    idx = v == y
    plt.scatter(PCA_data[idx, 0], PCA_data[idx, 1], label = i, s=100)
    # plt.scatter(X[idx, 0], X[idx, 2], label = i, s=4)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# %% Evaluation_index : Silohouette
  
# n_test = 20
# df_clusters, df_silhouette = plot_silhouette(X, n_test, PCA_data) #Silhouette
# df_silhouette_score = pd.concat([df_clusters, df_silhouette], axis=1) 
# df_silhouette_score.columns = ['cluster','silhouette']
# df_silhouette_score.to_csv("silhouette_"+file_name, index = None, encoding='euc-kr')


#%%: Evaluation_index: dunn index 

# dunn_list, cluster_no_list = dunn_index_test(X, 20)
# plt.figure()
# plt.plot(np.array(cluster_no_list), np.array(dunn_list),'o-')
# plt.title("Dunn_index")
# plt.show()

# df_dunn = pd.concat([pd.DataFrame(cluster_no_list), pd.DataFrame(dunn_list)], axis = 1) 
# df_dunn.columns = ['cluster','dunn']
# df_dunn.to_csv("dunn_"+file_name, index = None, encoding='euc-kr')

#%% Optimal cluster execution
# clustering_algorithms = Clustering_models(n_clusters)
# clster_pred = exectue_cluster(clustering_algorithms, n_clusters, PCA_data, label)

#test data 
# test_data = [16,1] #PCA 넣어서 행으로 추출해서 보내야함!
# model = KMeans(n_clusters=n_clusters)
model = AgglomerativeClustering(n_clusters=n_clusters)


result_priority = Recommend_test_data(PCA_data, test_data.values, y, n_clusters, model)
