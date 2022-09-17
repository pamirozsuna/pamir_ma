import pandas as pd
import numpy as np
from itertools import combinations
import graph_measures.functions as func
import networkx as nx
from itertools import permutations
from networkx.exception import NetworkXNoPath


class network:
    """Defines input as network
    :parameter pd.DataFrame that contains the adjacency matrix of the network, np.ndarray timecourse matrix
    TODO use absolute values or set negativ values to zero
    """
    def __init__(self, Adjacency_Matrix, tc=[]):
        assert isinstance(Adjacency_Matrix, (pd.DataFrame, np.ndarray)), "Input must be numpy.ndarray or panda.DataFrame"
        self.adj_mat=pd.DataFrame(Adjacency_Matrix)
        self.adj = Adjacency_Matrix
        self.nodes = list(self.adj_mat.index)

        if tc:
            assert isinstance(tc, np.ndarray), "Timecourse must be np.ndarray"
            self.time_course=pd.DataFrame(tc, index=self.nodes)
            self.cov_mat=func.covariance_mat(tc)
        else:
            self.time_course=None

    def Degree(self, node="all"):
        """
        Calculate the degree of each node in the network.
        :return n dimensional pd.Series with degrees of all nodes in the network
        """
        return self.adj_mat.sum(axis=1)-1

    def shortestpath(self):
        """
        Calculate the shortest path between all nodes in the network using Dijstrak Algorithm:
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
        :return Dictionary of two nxn dimensional pd.DataFrames with shortest path / shortest distance between all pairs of nodes in the network
        """
        inv_adj_mat=self.adj_mat.abs().pow(-1)                                                                          # Inverts adjacency matrix
        shortestdist_df=pd.DataFrame(np.zeros(inv_adj_mat.shape), columns=self.nodes, index=self.nodes)                 # Initialize Path matrix and distance matrix
        shortestpath_df=pd.DataFrame(np.empty(inv_adj_mat.shape, dtype=str), columns=self.nodes, index=self.nodes)
        counter=0
        for n in range(len(self.nodes)):
            node_set=pd.DataFrame({'Distance': np.full((len(self.nodes)-counter), np.inf),
                                   'Previous': ['']*(len(self.nodes)-counter), 'Path': ['']*(len(self.nodes)-counter)}, index=self.nodes[n:])
            node_set.loc[self.nodes[n], 'Distance'] = 0
            unvisited_nodes=self.nodes[n:]
            while unvisited_nodes != []:
                current=node_set.loc[unvisited_nodes,'Distance'].idxmin()    # Select node with minimal Distance of the unvisited nodes
                unvisited_nodes.remove(current)
                for k in self.nodes[n:]:
                    dist=node_set.loc[current, 'Distance'] + inv_adj_mat.loc[current, k]
                    if node_set.loc[k,'Distance'] > dist:
                        node_set.loc[k,'Distance'] = dist
                        node_set.loc[k,'Previous'] = current
            shortestdist_df.loc[n:,n]=node_set.loc[:,'Distance']
            shortestdist_df.loc[n, n:]=node_set.loc[:,'Distance']
            # Create Dataframe with string values for the shortest path between each pair of nodes
            for k in self.nodes[n:]:
                path=str(k)
                current=k
                while node_set.loc[current, 'Previous'] != '':
                    current=node_set.loc[current, 'Previous']
                    path=str(current)+'-'+path
                node_set.loc[k,'Path']=path
            shortestpath_df.loc[n:,n]=node_set.loc[:,'Path']
            shortestpath_df.loc[n,n:]=node_set.loc[:,'Path']
            counter += 1
        return {'Distance': shortestdist_df, 'Path': shortestpath_df}

    def Density(self):
        """
        The density for undirected graphs is
        The density is 0 for a graph without edges and 1 for a complete graph.

        :param graph: undirected weighted graph
        :return: density
        """
        wsum =np.sum(np.sum(self.adj_mat))
        return [np.divide(wsum,len(self.nodes)*(len(self.nodes)-1))]

    def Assortativity(self):
        """
        Compute degree assortativity of graph.

        Assortativity measures the similarity of connections
        in the graph with respect to the node degree.

        :param graph: undirected weighted graph
        :return: degree assortativity and peasrson assortativitity of graph
        """

        return [nx.degree_pearson_correlation_coefficient(nx.Graph(self.adj),weight='weight')]

    def Fiedler(self):
        """
        The Fiedler value, or algebraic connectivity, was introduced by Fiedler in 1973 and
        can be thought of as a measure of network robustness (Fiedler, 1973). The Fiedler value is equal
         to the second-smallest eigenvalue of the Laplacian matrix. The second smallest eigenvalue is used
         as it can be proven that the smallest eigenvalue of the Laplacian is always zero (Fiedler, 1973).
        The Laplacian matrix combines both degree information and connectivity information in the same matrix.

        :param graph: undirected weighted graph
        :return: second smallest eigen value
        """
        return [np.sort(nx.laplacian_spectrum(nx.Graph(self.adj_mat),weight='weight'))[1]]

    def num_triangles(self):
        """
        Calculate sum of triangles edge weights around each node in network
        :return: n dimensional pd.Series
        """
        triangles=pd.Series(np.zeros(len(self.nodes)), index=self.nodes)
        all_combinations=combinations(self.nodes, 3)        # Create list of all possible triangles
        abs_adj_mat = self.adj_mat.abs()
        sum_dict={}
        for combi in all_combinations:
            n1_n2=abs_adj_mat.loc[combi[0],combi[1]]        # Get path length between pairs in triangle combination
            n1_n3=abs_adj_mat.loc[combi[0],combi[2]]
            n2_n3=abs_adj_mat.loc[combi[1],combi[2]]
            sum_dict[combi]=(n1_n2+n1_n3+n2_n3)**(1/3)       # Calculate the triangle sum of the combination and save it in dictionary
        for node in self.nodes:
            triangles[node]=0.5*np.sum([sum_dict[s] for s in sum_dict if node in s])    # Sum all of the triangles that contain the node
        return triangles

    def char_path(self):
        """
        Calculate the characteristic path length of the network
        :return: Dictionary with average node distance np.array and characteristic path length np.float object
        """
        sum_shrtpath_df=self.shortestpath()['Distance'].sum(axis=1)             # Sums Shortest Path Dataframe along axis 1
        avg_shrtpath_node=np.divide(sum_shrtpath_df, len(self.nodes)-1)  # Divide each element in sum array by n-1 regions
        char_pathlength=np.sum(avg_shrtpath_node)/len(self.nodes)
        return {'node_avg_dist':avg_shrtpath_node, 'characteristic_path': char_pathlength}    # Calculate sum of the sum array and take the average

    def GlobalEfficiency(self):
        """
        Calculate the global efficiency of the network
        :return: np.float object
        """
        n = len(self.nodes)
        denom = n * (n - 1)
        try:
            if denom != 0:
                shortest_paths = dict(nx.all_pairs_dijkstra(nx.Graph(self.adj_mat), weight = 'weight'))
                g_eff = sum(1./shortest_paths[u][0][v] if shortest_paths[u][0][v] !=0 else 0 for u, v in permutations(nx.Graph(self.adj_mat), 2)) / denom
            else:
                g_eff = 0.0
        except KeyError:
                g_eff = 0.0
        return [np.asanyarray(g_eff)]

    def ClusteringCoefficient(self):
        """
        Calculate the cluster coefficient of the network
        :return: Dictionary of network cluster coefficient np.float object and ndim np.array of node cluster coefficients
        """
        triangles=np.multiply(np.array(self.num_triangles()), 2)
        degrees=np.array(self.Degree())
        excl_nodes=np.where(degrees < 2); triangles[excl_nodes]=0
        degrees=np.multiply(degrees, degrees-1)
        node_clust=np.divide(triangles,degrees)
        net_clust=(1/len(self.nodes))*np.sum(node_clust)
        #return {'node_cluster':pd.Series(node_clust, index=self.nodes), 'net_cluster':net_clust}
        return [net_clust]

    def Transitivity(self):
        """
        Calculate the transitivity of the network
        :return: np.float
        """
        triangles=np.sum(np.multiply(np.asarray(self.num_triangles()),2))     # Multiply sum of triangles with 2 and sum the array
        degrees=np.array(self.Degree())
        degrees=np.sum(np.multiply(degrees, degrees-1))
        return [np.divide(triangles, degrees)]

    def closeness_centrality(self):
        """
        Calculate the closeness centrality of each node in network
        :return: ndimensional pd.Series
        """
        node_avg_distance=self.char_path()['node_avg_dist']
        return pd.Series(np.power(node_avg_distance, -1), index=self.nodes)

    def betweenness_centrality(self):
        """
        Calculate the betweenness centrality of each node in network
        :return: ndimensional pd.Series
        """
        betw_centrality=pd.Series(np.zeros(len(self.nodes)), index=self.nodes)
        shortest_paths=self.shortestpath()['Path']

        for n in self.nodes:
            counter = 0
            mat=shortest_paths.drop(n, axis=0); mat=mat.drop(n, axis=1)  # Drops the nth column and the nth row.
            substr='-'+str(n)+'-'

            for c in mat.columns:
                for e in mat.loc[:c,c]:
                    if e.find(substr) != -1:
                        counter += 1
            betw_centrality.loc[n]=counter/((len(self.nodes)-1)*(len(self.nodes)-2))

        return betw_centrality

    def small_worldness(self, nrandnet=10, niter=10, seed=None, hqs=False, tc=[]):
        """
        Computes small worldness (sigma) of network
        :param: seed: float or integer which sets the seed for random network generation
                niter: int of number of iterations that should be done during network generation
                hqs: boolean value defines if hqs is used for random network generation
                tc: timecourse as np.ndarray for hqs algorithm
        :return:
        """
        import graph_measures.random_reference as randomnet
        random_clust_coeff=[]
        random_char_path=[]

        for i in range(nrandnet):

            if hqs:
                if tc: tc=self.time_course
                assert tc, "Timecourse not specified"
                random_net=randomnet.hqs_rand(tc)
            else:
                random_net=randomnet.rewired_rand(self.adj_mat, niter, seed)

            random_clust_coeff.append(random_net.clust_coeff()['net_cluster'])
            random_char_path.append(random_net.char_path()['characteristic_path'])

        random_clust_coeff=np.mean(random_clust_coeff)
        random_char_path=np.mean(random_char_path)

        sig_num=(self.clust_coeff()['net_cluster']/random_clust_coeff)
        sig_den=(self.char_path()['characteristic_path']/random_char_path)
        sigma=sig_num/sig_den
        return sigma


    def modularity(self):
        #TODO find algorithm to find modules in network
        return

