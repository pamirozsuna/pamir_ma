import pandas as pd
import numpy as np
import graph_measures.network as net
import graph_measures.functions as func

def hqs_rand(tc):
    """
    Returns a random network that is matched to the input networks covariance matrix.
    Using Hirschberger-Qi-Steuer Algorithm as cited in Zalesky 2012b

    :return: n x n dimensional pd.Dataframe
    TODO test code, control if input has to be is positive finite
    """
    if not isinstance(tc, (np.ndarray, pd.DataFrame)):
        raise ValueError('Timecourse has to be np.ndarray or pd.DataFrame')
    C = func.covariance_mat(tc)

    diag_sum = np.sum(np.diagonal(C))
    diag_len = len(np.diagonal(C))
    diag_mean = diag_sum / diag_len

    off_mean = (np.sum(C) - diag_sum) / (C.size - diag_len)
    off_var = C-off_mean
    np.fill_diagonal(off_var,0)                     # Sets diagonal values to zero
    off_var = np.sum(off_var ** 2)/(C.size-diag_len)

    m = max(2, (diag_mean ** 2 - off_mean ** 2 / off_var))
    mu = np.sqrt(off_mean / off_var)
    sigma = (-mu) ** 2 + np.sqrt(mu ** 4 + (off_var / m))
    X = np.random.normal(mu, sigma, C.size)

    random_covmat = np.matmul(X, X.T)
    covmat_diag=np.diagonal(random_covmat)
    inv_covmat_diag=np.diag(1/(covmat_diag))        # Invert the diagonal array and convert into matrix
    random_adjacency_mat=np.matmul((np.matmul(inv_covmat_diag, random_covmat)), inv_covmat_diag)
    random_net=net.network(random_adjacency_mat)
    return random_net

def rewired_rand(adjacency, niter=1, seed=None):
    #TODO write description, random seed? , double check with networkx
    num_nodes=adjacency.shape[0]
    node_list=list(adjacency.index)
    if seed: np.random.seed(seed)                   # Set seed to make random network replicable

    random_adj = np.array(adjacency)                # Sets up random network that is rewired in the following
    for r in range(niter):                          # Number rewiring iterations
        edge_list=[(i,j) for i in range(num_nodes) for j in range(num_nodes) if random_adj[i,j] != 0] # Create list of all edges
        num_edges=len(edge_list)/2
        rewired=[]
        n = 0
        s = 0
        while n<num_edges and s<num_edges/2:
            first_idx, second_idx = np.random.choice(len(edge_list), 2, replace=False) # Randomly choose two different edges in network
            i, j = edge_list[first_idx]                                     # Node indices in adjacency matrix of first edge
            n, m = edge_list[second_idx]                                    # Node indices in adjacency matrix of second edge

            if n in [i,j] or m in [i,j]:
                s += 1
                continue                                       # All nodes should be different
            if (i,n) in rewired or (j,m) in rewired:
                s += 1
                continue                                        # Check if the new edge was already rewired
            s = 0

            random_adj[i,n]=random_adj[i,j]             # Rewire i to n
            random_adj[n,i]=random_adj[i,j]             # Mirror edge to yield complete adjacency mat
            random_adj[j,m]=random_adj[n,m]             # Rewire j to m
            random_adj[m,j]=random_adj[n,m]

            edge_list=[edge for edge in edge_list if edge != (i,j) and edge != (j,i) and edge != (n,m) and edge != (m,n)]
            rewired.extend([(i,n),(j,m),(n,i),(m,j)])
            n += 2

    random_adj=pd.DataFrame(random_adj, index=node_list, columns=node_list) # Convert to DataFrame
    random_net=net.network(random_adj)  # Convert to network
    return random_net


