"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset.split('_')[0] == 'twitter':
    	adj, features = load_twitter_data(data_path, use_feats)[:2]
        #adj, features = load_twitter_dataV2(dataset, data_path)[:2] #if you want to use Brian's code
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']: # 2708, 19717
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc': # 1044
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport': # 3188
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset.split('_')[0] == 'twitter': # 583
            adj, features, labels = load_twitter_data(data_path, use_feats)
            #adj, features, labels = load_twitter_dataV2(dataset, data_path) #if you want to use Brian's code
            val_prop, test_prop = 0.25, 0.50
        elif dataset == 'lym':
            adj, features, labels = load_lym_data(data_path)
            val_prop, test_prop = 0.25, 0.25
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])

    print(features.shape)
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

def load_twitter_data(data_path, use_feats):
    
    with open(os.path.join(data_path, "all_twitter_ids.csv"), 'r') as f:
            all_twitter_ids = f.readlines()
    all_twitter_ids = all_twitter_ids[1:]
    all_twitter_ids = [x.rstrip('\n') for x in all_twitter_ids]
    all_twitter_ids = np.array(all_twitter_ids)
    
    idx_to_object = dict(enumerate(all_twitter_ids.flatten()))
    object_to_idx= {v: k for k, v in idx_to_object.items()}   
    
    adj = np.zeros((all_twitter_ids.shape[0], all_twitter_ids.shape[0]))
    
    names = ['favorite_list', 'friend_list', 'mention_list', 'reply_list', 'retweet_list']
    #names = ['friend_list']
    idx_counter = 0
    edges = []
    for name in names:
        with open(os.path.join(data_path, "{}.csv".format(name)), 'r') as f:
            all_edges = f.readlines()
        all_edges = all_edges[1:]
        for line in all_edges:
            n1, n2, _ = line.rstrip().split('\t')
            i = object_to_idx[n1]
            j = object_to_idx[n2]
            edges.append((i, j))            
    
    print(adj.shape)
    for i, j in edges:
        adj[i, j] = 1.  
        #adj[j, i] = 1. # comment this line for directed adjacency matrix
        
    data = pd.read_csv(os.path.join(data_path, "dict.csv"), sep='\t', dtype={"twitter_id":'str'})
    #data.drop(data[data['party'] == 'I'].index, inplace=True)
    data.replace({'party': {'D': 0, 'I': 0, 'R': 1}}, inplace=True)
    data = data[['twitter_id', 'party']]#.values
    
    labels = np.asarray([])
    for key in object_to_idx:
    	labels = np.concatenate((labels, data[data['twitter_id']==key]['party'].to_numpy()))
    	#labels.append(data[data['twitter_id']==key]['party'].value)
    
    #labels = data[:, 1]
        
    
    if use_feats == 1:
        user_features = np.load(os.path.join(data_path, 'features.npz'))
        features = np.concatenate(
            (user_features['description'], user_features['status']),
            axis=1)
    elif use_feats == 2:
        features = np.load(os.path.join(data_path, 'features.npz'))['description']
    elif use_feats == 3:
        features = np.load(os.path.join(data_path, 'features.npz'))['status']
    else:
        features = sp.eye(adj.shape[0])
    return sp.csr_matrix(adj), features, labels 

#for dataset use twitter_$type of edge$, so for example twitter_friend, twitter_mention
#type of edge is based on names of the data files
def load_twitter_dataV2(dataset, data_path):
    #resolve the data_path cause of my weird code
    data_path, _ = os.path.split(data_path)
    data_path = os.path.join(data_path, dataset.split('_')[0]) #this will fix the last directory to /twitter/

    dict_df = pd.read_csv(os.path.join(data_path, "dict.csv"), sep='\t', float_precision="round_trip",  dtype={"twitter_id":'str'})
    nodes = np.array(dict_df.index.values)

    edge_type = dataset.split('_')[1] #favorite, friend, mention, reply, or retweets
    edge_df= pd.read_csv(os.path.join(data_path, "{}_list.csv".format(edge_type)), sep='\t', dtype='str')#{"follower":'str', "followee":'str'})

    #create dictionaries, node ids are assigned as they are encountered in dict_df, so the first entry would have node id 0
    nid_to_tid = dict_df["twitter_id"].to_dict() #dict that converts from node id to twitter id
    tid_to_nid = {v: k for k, v in nid_to_tid.items()} #dict that converts from twitter id to node id

    #map twitter ids from original data to their node id
    #actor is the one doing the action, following, favoriting, liking, etc
    #recipient is the one receiving the action, so they get a new follower or someone likes their tweet, etc
    actor_map = edge_df.iloc[:,0].map(tid_to_nid)
    recipient_map = edge_df.iloc[:,1].map(tid_to_nid)

    #create edges np array for shape (#num edges x 2) where each row represents an edge
    edges =np.vstack((actor_map.to_numpy(), recipient_map.to_numpy())).T

    adj = np.zeros((nodes.shape[0], nodes.shape[0]))
    for edge in range(edges.shape[0]):
      actor = edges[edge, 0]
      recipient = edges[edge, 1]
      adj[actor, recipient] = 1

    #dict_df.drop(dict_df[dict_df['party'] == 'I'].index, inplace=True)
    dict_df.replace({'party': {'D': 0, 'I': 0, 'R': 1}}, inplace=True)
    features, labels = sp.eye(dict_df.shape[0]), dict_df['party'].values
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

