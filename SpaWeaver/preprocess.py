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


def read_h5ad(h5_path):
    adata = sc.read_h5ad(h5_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata

    
def read_Xenium(root_path):
    adata = sc.read_10x_h5(root_path + 'cell_feature_matrix.h5')
    adata.obs = pd.read_csv(root_path + 'cells.csv', index_col=0)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
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


def mt_qc(adata, prefix='MT-', pct=20):
    sc.pp.filter_genes(adata, min_cells=1)

    adata.var[prefix] = adata.var_names.str.startswith(prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=[prefix], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs[f'pct_counts_{prefix}'] < pct, :]
    adata = adata[:, ~adata.var[prefix]].copy()
    return adata

    
def read_HE_image(img_path, suffix='.ome.tif'):
    import tifffile as tiff

    scale = -1
    if suffix == '.ome.tif':
        import xml.etree.ElementTree as ET

        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        metadata = ome_tif.ome_metadata
        ome_tif.close()

        root = ET.fromstring(metadata)
        namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        pixels_element = root.find('.//ome:Pixels', namespace)
        if pixels_element is not None:
            pixels_attributes = pixels_element.attrib
            for attr, value in pixels_attributes.items():
                if attr == 'PhysicalSizeX' or attr == 'PhysicalSizeX':
                    scale = float(value)
                    break
    elif suffix == '.png' or suffix == '.jpg':
        image = Image.open(img_path)
        image_data = np.array(image)
    elif suffix == '.tif':
        ome_tif = tiff.TiffFile(img_path)
        image_data = ome_tif.asarray()
        ome_tif.close()
    else:
        print("Only support '.ome.tif', '.png' or '.jpg' file currently.")
    return image_data, scale


def register_physical_to_pixel(adata, transform_matrix, scale=1,
                               raw_key=['x_centroid', 'y_centroid'],
                               matrix_type='pixel2phsical',  # 'pixel2phsical'或者'physical2pixel'
                               prefix='image'):
    scale_old = np.sqrt(transform_matrix[0, 0] ** 2 + transform_matrix[0, 1] ** 2)
    scale = scale / scale_old
    transform_matrix = transform_matrix * scale
    transform_matrix[-1, -1] = 1

    if matrix_type == 'pixel2phsical':
        transform_matrix = np.linalg.inv(transform_matrix)

    x = adata.obs[raw_key[0]].values
    y = adata.obs[raw_key[1]].values
    ones = np.ones_like(x)
    coor_raw = np.vstack([x, y, ones])

    coor_new = (transform_matrix @ coor_raw)[:2, :]
    image_coor = np.round(coor_new).astype(int)
    adata.obsm[prefix + '_coor'] = image_coor.T
    adata.obs[prefix + '_col'] = image_coor[0]
    adata.obs[prefix + '_row'] = image_coor[1]
    return adata


def tiling_HE_patches(args, adata, img, key='image_coor', get_filter=False):  # iStar中说，单细胞大小约为8um*8um
    print('======================== Tiling HE patches for each single cells ===========================')
    if args.cell_diameter > 0:
        patch_radius = np.round(args.cell_diameter / args.scale / 2).astype(int)  # 映射到pixel
    else:
        patch_radius = int(args.resolution / 2.0)
        print("patch radius is ", patch_radius)
    outlier_cells1 = np.where(adata.obsm[key] < patch_radius)[0]
    outlier_cells2 = np.where(adata.obsm[key][:, 0] > (img.shape[1] - patch_radius))[0]
    outlier_cells3 = np.where(adata.obsm[key][:, 1] > (img.shape[0] - patch_radius))[0]
    outlier_cells = np.unique(np.hstack([outlier_cells1, outlier_cells2, outlier_cells3]))
    if len(outlier_cells) != 0:
        print('Remove the outlier cells, and Anndata file was reduced!')
        inlier_cells = set(np.arange(adata.n_obs)) - set(outlier_cells)
        adata = adata[list(inlier_cells)]
    he_patches = [0] * adata.n_obs
    adata.obsm[key] = adata.obsm[key].astype(int)
    for i in tqdm(range(adata.n_obs)):
        x, y = adata.obsm[key][i]
        he_patches[i] = torch.tensor(img[y - patch_radius: y + patch_radius, x - patch_radius:x + patch_radius])

    return torch.stack(he_patches, dim=0) / 255.0, adata


def extract_HE_patches_representation(args, he_patches, store_key=None, adata=None, skip_embedding=False):
    import torchvision.transforms as transforms
    from .utils import create_ImageEncoder

    if he_patches.dim() == 3:
        he_patches = he_patches.unsqueeze(0)  # 如果不是batch，则补充batch维度
    if he_patches.size(1) != 3:
        he_patches = he_patches.permute(0, 3, 1, 2)  # 通道维度放前面, (batch, channel, x, y)
    # he_patches = he_patches.float()

    print('====================== Extracting HE representations for each cell =========================')
    preprocess = transforms.Compose([transforms.Resize(224),
                                     # transforms.ToTensor(),
                                     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), )])

    representaions = []
    batch_num = int(np.ceil(he_patches.size(0) / args.img_batch_size))
    # batch_num = 12
    if not skip_embedding:
        model = create_ImageEncoder(args.image_encoder)
        model.to(args.device)
        model.eval()
        for i in tqdm(range(batch_num)):
            img_tensor = preprocess(
                he_patches[i * args.img_batch_size:min((i + 1) * args.img_batch_size, he_patches.size(0))].to(
                    args.device))
            with torch.no_grad():
                features = model(img_tensor).squeeze().detach().cpu().numpy()
                representaions.append(features)
            torch.cuda.empty_cache()
    else:
        for i in tqdm(range(batch_num)):
            img_tensor = preprocess(
                he_patches[i * args.img_batch_size:min((i + 1) * args.img_batch_size, he_patches.size(0))])
            representaions.append(img_tensor)

    representaions = np.vstack(representaions)
    if isinstance(store_key, str):
        adata.obsm[store_key] = representaions
    return representaions
    
    
def load_he_emb(adata, meta_root, tag):
    cell_filter = np.load(f'{meta_root}cell_filter/{tag}.npy', allow_pickle=True)
    adata = adata[cell_filter]
    adata.obsm['he'] = np.load(f'{meta_root}he_emb/{tag}.npy')
    return adata


def preprocess_rna(adata, selected_genes=None, target_sum=None, scale=False, n_hvg=-1):
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

    
def preprocess_protein(adata):
    def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)
    adata.X = np.apply_along_axis(protein_norm, 1, np.array(adata.X))
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