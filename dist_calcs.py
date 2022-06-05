# -*- coding: utf-8 -*-
"""
Spyder Editor

Script file to take in embedding file and adjacency matrix file in order to calculate distance between connected nodes
and link predictions.

"""

import logging
import numpy as np
import os
from scipy import sparse
from scipy.sparse import csr_matrix
from config import parser
import statistics

 
def minkowski_dot(x, y):
    first  = -1* np.dot(x[0],y[0])
    second = np.dot(x[1:],y[1:])
    inner_prod = np.sum([first,second])
    return inner_prod

#parent_dir = args.parent_dir
parent_path = "os.path.dirname(os.getcwd())"

#egcn_hgcn_dir = args.egcn_hgcn_dir
egcn_hgcn_path = "/Users/rosagarza/EGCNvsHGCN"

#embed_file = args.embed_file
embed_path = "logs/lp/2022_5_30/3/lp_Euclidean_embeddings.npy"

#adj_fille  = args.adj_file
adj_path = "logs/lp/2022_5_30/3/adj_matrix.npz"


#embeddings = os.path.join(egcn_hgcn_dir, embed_file)
#adj_matrix = os.path.join(egcn_hgcn_dir, adj_file)
embeddings = os.path.join(egcn_hgcn_path, embed_path)
adj_matrix = os.path.join(egcn_hgcn_path, adj_path)

model = "GCN"
manifold = "Euclidean"
task = "lp"
c    = 1.0


emb_data = np.load(embeddings)
adj_data = sparse.load_npz(adj_matrix)

adj_1d = csr_matrix(adj_data)
#print("Sparse matrix: \n",S)
adj_2d = adj_1d.todense()
print("Dense matrix: \n", adj_2d)

dist = {}
if(manifold=="Euclidean"):
    #dist_euclid = {}
    # For Euclidean distances -- https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity
    for node in range(len(emb_data)):
        for adj_node in range(adj_2d[node,:].shape[1]):
            if(adj_2d[node,adj_node]!=1.0):
                # Euclidean distances
               dist[(node,adj_node)] = np.sqrt((np.square(emb_data[node]-emb_data[adj_node])).sum())
    
# Hyperbolic spaces - distances between 2 points
if(manifold == "Hyperboloid"):
    #dist_hyper = {}
    # Manifold Hyperboloid class -- states c = 1/K, so K= 1/c
    # Formula from: https://drive.google.com/drive/folders/1wu8m5sKvtJJVzVQPVtKjwYiQH5btpMiQ
    for node in range(len(emb_data)):
        for adj_node in range(adj_2d[node,:].shape[1]):
            if(adj_2d[node,adj_node]!=1.0):
                # Hyperboloid distances  
                K = 1.0/c        
                mink_dot = minkowski_dot(emb_data[node],emb_data[adj_node])
                theta    = -mink_dot/K
                hyp_dist   = np.sqrt(K)*np.arccosh(theta)
                dist[(node,adj_node)] = hyp_dist

     
# poincare -- https://dawn.cs.stanford.edu/2018/03/19/hyperbolics
if(manifold == "PoincareBall"):
    for node in range(len(emb_data)):
        for adj_node in range(adj_2d[node,:].shape[1]):
            if(adj_2d[node,adj_node]!=1.0):
                # Poincare distances
                numer     =  np.square(np.linalg.norm(emb_data[node]-emb_data[adj_node]))
                left_den  =  np.square(np.linalg.norm(1-emb_data[node]))
                right_den =  np.square(np.linalg.norm(1-emb_data[adj_node]))
                den       = left_den * right_den
                dist[(node,adj_node)] = np.arccosh(1+ 2*(numer/den))


mean_dist = np.mean(list(dist.values()))         
max_dist  = max(dist.values())


min_val = 1000
for item in dist:
    dupes = all(i == list(item)[0] for i in list(item))
    if(dist[item]< min_val and not(dupes)):
        min_val = dist[item]
        # sanity that min of 0 isn't 2 nodes that are same
        #tup = item

logging.info(f"Model: {manifold}")
logging.info(f"Task: {task}")
logging.info(f"Minimum Distance: {min_val}")
logging.info(f"Maximum Distance: {max_dist}")
logging.info(f"Average Distance: {mean_dist}")
#np.save(os.path.join(save_dir, '{}_embeddings.npy'.format(args.task+"_"+args.manifold)), best_emb.cpu().detach().numpy())

           


#print(emb_data)
#print(adj_data)
#if __name__ == '__main__':
#    args = parser.parse_args()
#    dist_calcs(args)
