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
    """Read a 10x Genomics Visium HD dataset and its spatial metadata.

    Parameters
    ----------
    root_path : str
        Dataset directory containing ``filtered_feature_bc_matrix.h5`` and a
        ``spatial`` subdirectory with ``tissue_positions.parquet`` and
        ``scalefactors_json.json``.

    Returns
    -------
    anndata.AnnData
        Expression data with unique observation and feature names. Pixel
        coordinates are stored in ``.obsm['image_coor']``, physical spatial
        coordinates in ``.obsm['spatial']``, and array coordinates in
        ``.obs['array_row']`` and ``.obs['array_col']``.

    Notes
    -----
    Physical coordinates are computed by multiplying full-resolution pixel
    coordinates by ``microns_per_pixel`` from ``scalefactors_json.json``.
    """
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
    """Filter low-quality cells and mitochondrial genes from expression data.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object containing gene expression in ``.X``.
    prefix : str, optional
        Prefix used to identify mitochondrial genes in ``adata.var_names``.
    pct : float, optional
        Maximum permitted percentage of counts assigned to mitochondrial
        genes. Cells at or above this threshold are removed.

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object with mitochondrial genes removed.

    Notes
    -----
    The function first removes genes expressed in fewer than one cell and
    writes Scanpy quality-control metrics to ``adata.obs`` before filtering.
    """
    sc.pp.filter_genes(adata, min_cells=1)

    adata.var[prefix] = adata.var_names.str.startswith(prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=[prefix], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs[f'pct_counts_{prefix}'] < pct, :]
    adata = adata[:, ~adata.var[prefix]].copy()
    return adata

    
def read_HE_image(img_path, suffix='.ome.tif'):
    """Read an H&E image and its physical pixel size when available.

    Parameters
    ----------
    img_path : str or pathlib.Path
        Path to the H&E image file.
    suffix : {'.ome.tif', '.png', '.jpg', '.tif'}, optional
        Image format used to select the reader. This value is not inferred
        from ``img_path``. Defaults to ``'.ome.tif'``.

    Returns
    -------
    image_data : numpy.ndarray
        Image array with the shape and channel order returned by the reader.
    scale : float or int
        Value of ``PhysicalSizeX`` from OME metadata, or ``-1`` for other
        formats or when that attribute is absent. Units are those of the
        metadata; no unit conversion is performed.

    Notes
    -----
    OME-TIFF images must contain parseable OME metadata. Pixel size is read
    using the OME 2016-06 XML namespace. Image channels and axes are returned
    unchanged, so callers should check their layout before extracting patches.
    """
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
    """Transform physical cell coordinates into integer image coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing physical coordinates in ``.obs``.
    transform_matrix : numpy.ndarray of shape (3, 3)
        Homogeneous coordinate transformation matrix. Its direction is
        specified by ``matrix_type``.
    scale : float, optional
        Scale used to rescale the transformation matrix, defaulting to ``1``.
        The matrix is multiplied by ``scale / scale_old``, where
        ``scale_old`` is the Euclidean norm of its first two entries in the
        first row. The bottom-right entry is then reset to ``1``.
    raw_key : sequence of str, optional
        Two ``.obs`` column names containing physical x and y coordinates,
        respectively. Defaults to ``['x_centroid', 'y_centroid']``.
    matrix_type : {'pixel2phsical', 'physical2pixel'}, optional
        Direction of the supplied matrix. ``'pixel2phsical'`` (the default,
        retaining the API's spelling) inverts the rescaled matrix before
        applying it. ``'physical2pixel'`` applies it directly.
    prefix : str, optional
        Prefix for the output keys, defaulting to ``'image'``.

    Returns
    -------
    anndata.AnnData
        The input object, updated in place. Rounded integer coordinates in
        x, y order are stored in ``.obsm[prefix + '_coor']``. Columns and
        rows are also stored in ``.obs[prefix + '_col']`` and
        ``.obs[prefix + '_row']``, respectively.
    """
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
    """Extract square H&E image patches centered on cell coordinates.

    Parameters
    ----------
    args : argparse.Namespace or object
        Configuration with ``cell_diameter`` and either ``scale`` or
        ``resolution`` attributes. When ``cell_diameter > 0``, the patch
        radius in pixels is ``round(cell_diameter / scale / 2)``; diameter
        and physical pixel size must use the same units. Otherwise, the
        radius is ``int(resolution / 2)``.
    adata : anndata.AnnData
        AnnData object with image coordinates in ``.obsm[key]``, arranged
        as x (column), y (row) for each observation.
    img : numpy.ndarray of shape (height, width, channels)
        H&E image with pixel values on the 0--255 scale.
    key : str, optional
        Key in ``adata.obsm`` containing image coordinates. Defaults to
        ``'image_coor'``.
    get_filter : bool, optional
        Currently unused. Neither value adds a filter mask to the return
        values.

    Returns
    -------
    he_patches : torch.Tensor
        Stacked image patches with shape
        ``(n_retained_cells, 2 * patch_radius, 2 * patch_radius, channels)``.
        Pixel values are divided by ``255.0``.
    adata : anndata.AnnData
        AnnData object after removing cells whose patches would cross the
        image boundaries. Its observation order matches ``he_patches``, and
        ``.obsm[key]`` is converted to integer coordinates.

    Notes
    -----
    Use the returned AnnData object to keep observations aligned with the
    extracted patches. At least one cell must remain after boundary filtering.
    """
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
    """Extract image encoder representations from H&E patches in batches.

    Parameters
    ----------
    args : argparse.Namespace or object
        Configuration with ``img_batch_size`` (patches per batch),
        ``image_encoder`` (model name accepted by
        :func:`SpaWeaver.utils.create_ImageEncoder`), and ``device`` (PyTorch
        device). Only ``img_batch_size`` is used when ``skip_embedding=True``.
    he_patches : torch.Tensor
        Floating-point RGB patches with values in [0, 1], shaped
        ``(n_patches, height, width, 3)`` or
        ``(n_patches, 3, height, width)``. A single three-dimensional patch
        is also accepted. Use a CPU tensor when ``skip_embedding=True``.
    store_key : str or None, optional
        If a string, also store the returned array in ``adata.obsm[store_key]``.
        Defaults to ``None``, which disables storage.
    adata : anndata.AnnData or None, optional
        AnnData object aligned with the input patches. Required when
        ``store_key`` is a string; its observation count must match the
        number of patches.
    skip_embedding : bool, optional
        If ``True``, return resized and normalized image arrays without
        creating or running an image encoder. Defaults to ``False``.

    Returns
    -------
    numpy.ndarray
        Image features, typically shaped ``(n_patches, n_features)``, where
        the feature dimension depends on the encoder. When
        ``skip_embedding=True``, returns channel-first image arrays shaped
        ``(n_patches, 3, resized_height, resized_width)`` instead.

    Notes
    -----
    Patches are converted to channel-first layout, resized so that the
    shorter edge is 224 pixels, and normalized using ImageNet channel means
    ``(0.485, 0.456, 0.406)`` and standard deviations
    ``(0.229, 0.224, 0.225)``. Encoder inference runs in evaluation mode
    without gradient computation.
    """
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
    """Load precomputed H&E embeddings and apply their cell filter.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object whose observations are aligned with the stored cell
        filter.
    meta_root : str
        Directory prefix containing ``cell_filter`` and ``he_emb`` folders.
        Include a trailing path separator when necessary.
    tag : str
        Dataset identifier used to locate ``cell_filter/{tag}.npy`` and
        ``he_emb/{tag}.npy`` below ``meta_root``.

    Returns
    -------
    anndata.AnnData
        Cell-filtered AnnData object with the loaded H&E embedding stored in
        ``.obsm['he']``.
    """
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
    """Build a sparse neighborhood graph from coordinates or feature vectors.

    Parameters
    ----------
    x : numpy.ndarray of shape (n_nodes, n_features)
        Coordinates or feature vectors used to find neighboring nodes.
    weighted : bool or str, optional
        ``False`` (the default) assigns unit weights. ``'reciprocal'`` uses
        inverse distances. ``'gaussian'`` or any other truthy value uses the
        exponential weighting formula described in Notes. Use ``False``,
        not the string ``'none'``, for an unweighted graph.
    symmetric : bool, optional
        If ``True``, symmetrize the adjacency by taking the larger weight
        for each pair of opposite edges before normalization. Defaults to
        ``False``.
    graph_type : {'radius', 'knn'}, optional
        Neighborhood rule. ``'radius'`` (the default) connects nodes within
        ``radius``; ``'knn'`` queries ``num_neighbors`` nearest neighbors.
    metric : {'euclidean', 'cosine'}, optional
        Distance mode, defaulting to ``'euclidean'``. ``'cosine'`` first
        L2-normalizes input rows; neighbor searches still use Euclidean
        distance on those rows. For weighted graphs, the resulting
        distances are converted to cosine distances as ``d ** 2 / 2``.
    self_loop : bool, optional
        Whether to retain diagonal edges from the neighbor query. If
        ``False``, diagonal weights are set to zero. Defaults to ``True``.
    radius : float, optional
        Search radius for ``graph_type='radius'``, defaulting to ``50``.
        With ``metric='cosine'``, this is a Euclidean radius on normalized
        vectors, not a cosine-distance threshold.
    num_neighbors : int, optional
        Number of neighbors queried for ``graph_type='knn'``, defaulting to
        ``50``. The query includes the node itself before optional diagonal
        removal. Must not exceed the number of nodes.
    apply_normalize : {'none', 'row', 'col', 'both', 'gcn', 'bipart_gcn', 'hpnn'}, optional
        Normalization passed to ``normalize_graph`` after diagonal removal
        and symmetrization. Defaults to ``'none'``.
    sigma : float, optional
        Scale parameter in the exponential weighting formula, defaulting
        to ``0.01``. Unused for unit or reciprocal weights.
    return_type : {'csr', 'coo'}, optional
        SciPy sparse output format when ``device`` is ``None``. Defaults
        to ``'csr'``.
    device : str, torch.device or None, optional
        If provided, convert the adjacency to a float32 PyTorch sparse COO
        tensor on this device. Defaults to ``None``.

    Returns
    -------
    scipy.sparse.csr_matrix or scipy.sparse.coo_matrix or torch.Tensor
        Adjacency of shape ``(n_nodes, n_nodes)``. Rows and columns follow
        the order of ``x``. The output type is controlled by ``return_type``
        and ``device``.

    Notes
    -----
    For a weighting distance ``d``, the implemented exponential weight is
    ``exp(-(d ** 2) * sigma ** 2 / 2) / (2 * pi * sigma ** 2)``.
    Reciprocal weights do not guard against zero distances, including
    self-distances and coincident points.
    """
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
