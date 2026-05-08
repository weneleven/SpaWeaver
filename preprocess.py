import json
import itertools
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from PIL import Image, ImageFile
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
from .utils import sparse_mx_to_torch_sparse_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def read_Xenium(h5_path, obs_path):
    adata = sc.read_10x_h5(h5_path)
    adata.obs = pd.read_csv(obs_path, index_col=0)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    return adata


def preprocess_protein(adata):
    def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    adata.X = np.apply_along_axis(protein_norm, 1, np.array(adata.X))
    return adata


def read_h5ad(h5_path):
    adata = sc.read_h5ad(h5_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def read_VisiumHD(root_path):
    adata = sc.read_10x_h5(root_path + 'filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    spatial_path = root_path + 'spatial/tissue_positions.parquet'
    spatial = pd.read_parquet(spatial_path)
    spatial.index = spatial['barcode'].astype(str)
    spatial = spatial.loc[adata.obs_names]

    adata.obsm['image_coor'] = spatial[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
    adata.obs['array_row'], adata.obs['array_col'] = spatial['array_row'].values, spatial['array_col'].values

    scale_file = root_path + 'spatial/scalefactors_json.json'
    with open(scale_file, 'r') as f:
        data = json.load(f)
    adata.obsm['spatial'] = adata.obsm['image_coor'] * data['microns_per_pixel']
    return adata


def load_he_emb(adata, meta_root, tag):
    cell_filter = np.load(f'{meta_root}cell_filter/{tag}.npy', allow_pickle=True)
    adata = adata[cell_filter]
    adata.obsm['he'] = np.load(f'{meta_root}he_emb/{tag}.npy')
    return adata


def preprocess_adata(adata, selected_genes=None, target_sum=None, scale=False, n_hvg=-1):
    adata.layers['raw'] = adata.X.copy()
    if n_hvg > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")

    if scale:
        gene_min = adata.X.min(0)
        gene_max = adata.X.max(0)
        adata.var['min'] = gene_min
        adata.var['max'] = gene_max
        adata.X = adata.X - gene_min
        adata.X = adata.X / (gene_max - gene_min + 1e-12)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
        sc.pp.log1p(adata)
    
    if selected_genes is not None:
        var_name_set = set(adata.var_names)
        selected_genes = [gene for gene in selected_genes if gene in var_name_set]
        adata = adata[:, selected_genes].copy()
    elif n_hvg > 0:
        adata = adata[:, adata.var['highly_variable']].copy()
    
    if isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.todense().A
    return adata


def quality_control(adata, platform='VisiumHD', filter_var=True, filter_obs=True, qt_threshold=[0.05, 0.95], return_filters=False):
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-') 
    adata.var['ribo'] = adata.var_names.str.upper().str.startswith(("RPS","RPL"))
    var_filter = (adata.var['mt'] == False) & (adata.var['ribo'] == False)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata.obs['log10GenesPerUMI'] = np.log10(adata.obs['n_genes_by_counts']) / np.log10(adata.obs['total_counts'])
    if filter_var:
        adata = adata[:, var_filter]
    
    if platform == 'VisiumHD':
        obs_filter = (adata.obs['log10GenesPerUMI'] > 0.9) & (adata.obs['log10GenesPerUMI'] < 0.99)
        min_counts = int(np.quantile(adata.obs['total_counts'], qt_threshold[0]))
        max_counts = int(np.quantile(adata.obs['total_counts'], qt_threshold[1]))
        obs_filter = obs_filter & (adata.obs['total_counts'] > min_counts) & (adata.obs['total_counts'] < max_counts) & (adata.obs['pct_counts_mt'] < 30)
    else:
        obs_filter = (adata.obs['n_genes_by_counts'] > 50) & (adata.obs['pct_counts_mt'] < 38)
    if filter_obs:
        adata = adata[obs_filter]
    if return_filters:
        return adata, var_filter, obs_filter
    return adata
    

def build_graph(x, weighted=False, symmetric=False, graph_type='radius', metric='euclidean', self_loop=True,
                radius=50, num_neighbors=50, apply_normalize='none', sigma=0.01, return_type='csr', device=None):
    '''
    graph_type: str,    'radius' will connect the nodes within the radius(50 by default),
                        'knn' will connect the num_neighbors(50 by default) nearest neighbors

    weighted:   str,    'reciprocal' will lead to calculate the reciprocal of the distance as a weight
                        'gaussian' will lead to calculate the gaussian kernel as a weight, sigma is 1.5 by default
                        'none' will generate a binary adj

    symmetric:  bool    False will directly return the adj
                        True will makes adj[i, j] = adj[j, i]
    '''
    metric = metric.lower()
    apply_normalize = apply_normalize.lower()
    graph_type = graph_type.lower()

    if metric == 'cosine':
        x = normalize(x, norm='l2')

    tree = BallTree(x)
    if graph_type == 'radius':
        tail_list, distances = tree.query_radius(x, r=radius, return_distance=True)
    elif graph_type == 'knn':
        distances, tail_list = tree.query(x, k=num_neighbors)

    head_list = []
    head_list = [head_list + [i] * len(tail_list[i]) for i in range(len(tail_list))]
    head_list = list(itertools.chain.from_iterable(head_list))
    tail_list = list(itertools.chain.from_iterable(tail_list))

    if not weighted:
        distances = np.ones_like(head_list)
    else:
        distances = np.array(list(itertools.chain.from_iterable(list(distances))))
        if metric == 'cosine':
            distances = (distances * distances) / 2
        if weighted == 'reciprocal':
            distances = 1 / distances
        else:
            distances = np.exp(-(distances ** 2) / 2 * sigma * sigma) / (2 * np.pi * sigma * sigma)

    adj = sp.coo_matrix((distances, (head_list, tail_list)), shape=(x.shape[0], x.shape[0]))  # 用稀疏矩阵构建，方便后续计算

    if not self_loop:
        adj = adj.tocsr()
        adj.setdiag(0)

    if symmetric:
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if apply_normalize != 'none':
        adj = normalize_graph(adj, edge_weight=None, norm_type=apply_normalize)

    if return_type == 'coo':
        if not isinstance(adj, sp.coo_matrix):
            adj = adj.tocoo()
    elif return_type == 'csr':
        if not isinstance(adj, sp.csr_matrix):
            adj = adj.tocsr()

    if device is not None:
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    return adj


def normalize_graph(adj, edge_weight=None, norm_type='gcn', **kwargs):
    norm_type = norm_type.lower()

    if norm_type == 'row':
        normalization_factors = sp.csr_matrix(1.0 / (adj.sum(1) + 1e-6))  # 行归一化
        adj = adj.multiply(normalization_factors)
    elif norm_type == 'col':
        normalization_factors = sp.csr_matrix(1.0 / (adj.sum(0) + 1e-6))  # 列归一化
        adj = adj.multiply(normalization_factors)
    elif norm_type == 'both':
        normalization_factors1 = sp.csr_matrix(1.0 / (adj.sum(0) + 1e-6))  # 列归一化
        normalization_factors2 = sp.csr_matrix(1.0 / (adj.sum(1) + 1e-6))  # 行归一化
        adj = adj.multiply(normalization_factors1)
        adj = adj.multiply(normalization_factors2)
    elif norm_type == 'gcn':
        D = np.squeeze(adj.sum(1).A)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # 防止除以 0

        D_mat = sp.diags(D_inv_sqrt, format='csr')
        adj = D_mat @ adj @ D_mat
    elif norm_type == 'bipart_gcn':
        row_sum = np.array(adj.sum(1)).flatten()  # shape: (A,)
        col_sum = np.array(adj.sum(0)).flatten()  # shape: (B,)
        row_inv_sqrt = np.power(row_sum, -0.5)
        col_inv_sqrt = np.power(col_sum, -0.5)
        row_inv_sqrt[np.isinf(row_inv_sqrt)] = 0.0
        col_inv_sqrt[np.isinf(col_inv_sqrt)] = 0.0

        D_r_inv = sp.diags(row_inv_sqrt)
        D_c_inv = sp.diags(col_inv_sqrt)

        adj = D_r_inv @ adj @ D_c_inv
    elif norm_type == 'hpnn':
        DE = np.squeeze(adj.sum(0).A)
        DV = np.squeeze(adj.sum(1).A)
        DE = sp.diags(np.power(DE.astype(float), -1), offsets=0, format='csr')
        DV = sp.diags(np.power(DV.astype(float), -0.5), offsets=0, format='csr')
        if edge_weight != None:
            W = sp.diags(np.squeeze(edge_weight), offsets=0, format='csr')
        else:
            W = sp.diags(np.ones(shape=(adj.shape[1])), offsets=0, format='csr')
        adj = DV @ adj @ W @ DE @ adj.T @ DV
    return adj