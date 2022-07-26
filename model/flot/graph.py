import torch


class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size 点云的batch 和 n
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points (a-b)**2 = a**2 + b**2 - 2ab
        # dm (b,n,1)
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        # dm (b,n,n)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        # torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3
        # dm(b,n,n)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        # 在dm最后一维度排序 并得到由小到大索引 argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # 找到索引后 提取出最小距离的前nb_nei个邻居点
        # nei(b*n,nb_nei)
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        # ？enn 有效的邻居点的个数
        # enn (int)
        effective_nb_neighbors = neighbors.shape[-1]
        # nei(b,n,nb_nei)
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        # torch.arange(start=1.0,end=6.0)的结果不包括end idx=(tensor)[0,1,2...nb_p-1] (1,nb_p)
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # idx (enn, nb_p)
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph
