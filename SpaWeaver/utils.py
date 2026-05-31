import os
import timm
import random
import socket

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.distributed as dist
import torchvision.models as models
from tqdm import tqdm
from torch import optim as optim
from scipy.signal import periodogram
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import r2_score


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def structural_similarity_on_graph_data(x, y, adj, K1=0.01, K2=0.03, alpha=1, beta=1, gamma=1, sigma=1.5,
                                        use_sample_covariance=True):
    assert x.shape == y.shape

    K3 = K2 / np.sqrt(2)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if K3 < 0:
        raise ValueError("K3 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    R = x.max() - x.min()
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = (K3 * R) ** 2

    num_neighbor_list = adj.getnnz(axis=1)
    if use_sample_covariance:
        cov_norm = num_neighbor_list / (num_neighbor_list - 1 + 1e-6)  # 计算方差的norm
    else:
        cov_norm = 1 / (num_neighbor_list + 1e-6)
    cov_norm = cov_norm[:, np.newaxis]

    ux = adj @ x
    uy = adj @ y
    uxx = adj @ (x * x)
    uyy = adj @ (y * y)
    uxy = adj @ (x * y)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    A1 = 2 * ux * uy + C1
    A2 = 2 * np.sqrt(np.clip(vx * vy, 0, None)) + C2
    A3 = vxy + C3
    B1 = ux * ux + uy * uy + C1
    B2 = vx + vy + C2
    B3 = np.sqrt(np.clip(vx * vy, 0, None)) + C3
    S = (A1 / B1) ** alpha * (A2 / B2) ** beta * (A3 / B3) ** gamma
    return S.mean(0)


def compute_metrics(x, x_prime, metric='cosine_similarity', reduce='median', graph=None):
    metric = metric.lower()
    if metric == 'cosine_similarity':
        dot_product = np.sum(x_prime * x, axis=0)
        norm1 = np.linalg.norm(x_prime, axis=0)
        norm2 = np.linalg.norm(x, axis=0)
        metric = dot_product / (norm1 * norm2 + 1e-6)
    elif metric == 'rmse':
        mse = np.mean((x_prime - x) ** 2, axis=0)
        metric = np.sqrt(mse)
    elif metric == 'pcc':
        x_center = x - np.mean(x, axis=0)
        y_center = x_prime - np.mean(x_prime, axis=0)
        denominator = np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center, axis=0))
        metric = np.sum(x_center * y_center, axis=0) / (denominator + 1e-6)
    elif metric == 'ssim':
        print("x shape is ", x.shape[0])
        if x.shape[0] < 200000:
            print("cell number is less than 200000")
            metric = structural_similarity_on_graph_data(x, x_prime, graph)
        else:
            print("cell number is greater than 200000")
            idx_list = list(range(0, x.shape[0]))
            random.shuffle(idx_list)
            batch_size = 200000
            batch_num = int(np.ceil(x.shape[0] / batch_size))
            batch_size = int(np.ceil(x.shape[0] / batch_num))
            ssim_sum = np.zeros(x.shape[-1])
            print('To avoid memory overflow, the data is splited into ' + str(batch_size) + ' cells batches.')
            for i in tqdm(range(batch_num)):
                tgt_cells = idx_list[i * batch_size: min((i + 1) * batch_size, x.shape[0])]
                tgt_cells_potential = graph[tgt_cells].tocoo().col
                tgt_cells = list(set(tgt_cells).union(set(tgt_cells_potential)))
                metric = structural_similarity_on_graph_data(x[tgt_cells], x_prime[tgt_cells],
                                                             graph[tgt_cells][:, tgt_cells])
                ssim_sum = ssim_sum + metric
            metric = ssim_sum / batch_num
    elif metric == 'cmd':
        x = x + np.random.normal(0, 1e-8, x.shape)
        x_prime = x_prime + np.random.normal(0, 1e-8, x_prime.shape)

        if x.shape[1] < 10000:
            corr_pred = np.corrcoef(x_prime, dtype=np.float32, rowvar=0)
            corr_true = np.corrcoef(x, dtype=np.float32, rowvar=0)

            numerator = np.trace(corr_pred.dot(corr_true))
            denominator = np.linalg.norm(corr_pred, 'fro') * np.linalg.norm(corr_true, 'fro')

            metric = 1 - numerator / (denominator + 1e-8)
        else:
            idx_list = list(range(0, x.shape[1]))
            random.shuffle(idx_list)
            batch_size = 10000
            batch_num = int(np.ceil(x.shape[1] / batch_size))
            batch_size = int(np.ceil(x.shape[1] / batch_num))
            cmd_list = []
            print('To avoid memory overflow, the data is splited into ' + str(batch_size) + ' cells batches.')
            for i in tqdm(range(batch_num)):
                tgt_cells = idx_list[i * batch_size: min((i + 1) * batch_size, x.shape[1])]
                corr_pred = np.corrcoef(x_prime[:, tgt_cells], dtype=np.float32, rowvar=0)
                corr_true = np.corrcoef(x[:, tgt_cells], dtype=np.float32, rowvar=0)
                numerator = np.trace(corr_pred.dot(corr_true))
                denominator = np.linalg.norm(corr_pred, 'fro') * np.linalg.norm(corr_true, 'fro')
                metric = 1 - numerator / (denominator + 1e-8)
                cmd_list.append(metric)
            metric = np.array(cmd_list)
    else:
        print('Not implemented!')
        return np.nan, np.nan

    if reduce == 'mean':
        metric_reduce = metric.mean()
    elif reduce == 'sum':
        metric_reduce = metric.sum()
    elif reduce == 'median':
        metric_reduce = np.median(metric)
    return metric, metric_reduce


def compute_MoransI(adata, adj, store_key=None):
    n = adata.n_obs
    x_bar = np.mean(adata.X, axis=0)
    x = adata.X - x_bar

    numerator = np.sum(((adj @ x).A)*(x.A), axis=0)
    denominator = np.sum(x.A**2, axis=0)
    MoransI = (n / np.sum(adj)) * (numerator / (denominator + 1e-6))

    if isinstance(store_key, str):
        adata.var[store_key] = MoransI
    return MoransI


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    if isinstance(model, list):
        parameters = []
        for each in model:
            parameters.append({'params': each.parameters()})
    else:
        parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def find_free_port(start_port=29500, max_attempts=100):
    """
    查找一个空闲的端口
    
    参数:
        start_port: int, 默认29500
            起始端口号
        max_attempts: int, 默认100
            最大尝试次数
    
    返回:
        int: 找到的空闲端口号
    """
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                print('Using port: ' + str(port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts starting from {start_port}")


def setup(rank: int, world_size: int, backend = None) -> None:
    """
    Initialize torch.distributed process group for DDP training.

    Expected usage (as in your scripts):
      - launched via `torch.multiprocessing.spawn(..., nprocs=world_size)`
      - environment has MASTER_ADDR / MASTER_PORT set (init_method='env://')

    Args:
        rank: process rank (also used as local GPU id in single-node training)
        world_size: total number of processes
        backend: optional override ('nccl' or 'gloo'); auto-select if None
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    # Safe to call multiple times (e.g., nested utilities) as long as it's initialized once.
    if dist.is_initialized():
        return

    use_cuda = torch.cuda.is_available()
    if backend is None:
        backend = "nccl" if use_cuda else "gloo"

    # In single-node multi-GPU, rank maps to local GPU id (as used in your scripts).
    if use_cuda:
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, init_method="env://")


def cleanup() -> None:
    """Destroy torch.distributed process group (if initialized)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_ImageEncoder(model_name='resnet50', pretrained=True, frozen=True):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(pretrained=True)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(pretrained=True)
    elif model_name == 'vit_l_32':
        model = models.vit_l_32(pretrained=True)
    elif model_name == 'vit_h_14':
        model = models.vit_h_14(pretrained=True)
    elif model_name == 'uni':
        local_dir = "./UNI_weights/"
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    elif model_name == 'conch':
        model = None
    else:
        assert False

    if frozen:
        model.eval()
    return model


def create_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'prelu':
        return nn.PReLU()
    else:
        return None


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device='cpu'):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def re_features(adj, features, K, device='cpu'):
    if not torch.is_tensor(features):
        features = torch.Tensor(features).to(device)

    if not torch.is_tensor(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj, device=device)

    nodes_features = torch.empty((features.shape[0], K + 1, features.shape[1]), device=device, dtype=features.dtype)

    nodes_features[:, 0, :] = features
    x = features.clone()
    for i in range(K):
        x = torch.spmm(adj, x) if adj.is_sparse else adj @ x
        nodes_features[:, i + 1, :] = x
    nodes_features = nodes_features.detach().cpu().numpy()
    torch.cuda.empty_cache()
    return nodes_features

def re_features_np(adj, features, K):
    nodes_features = np.empty((features.shape[0], 1, K + 1, features.shape[1]), dtype=features.dtype,)

    nodes_features[:, 0, 0, :] = features
    x = features.copy()
    for i in range(K):
        x = adj @ x
        nodes_features[:, 0, i + 1, :] = x

    return nodes_features.squeeze()


def simonson_vHE(dapi_image, eosin_image):
    
    def createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin):
        new_image = np.empty([dapi_image.shape[0], dapi_image.shape[1], 3])
        new_image[:,:,0] = background[0] + (1 - background[0]) * np.exp(- k1 * beta_DAPI[0] * dapi_image - k2 * beta_eosin[0] * eosin_image)
        new_image[:,:,1] = background[1] + (1 - background[1]) * np.exp(- k1 * beta_DAPI[1] * dapi_image - k2 * beta_eosin[1] * eosin_image)
        new_image[:,:,2] = background[2] + (1 - background[2]) * np.exp(- k1 * beta_DAPI[2] * dapi_image - k2 * beta_eosin[2] * eosin_image)
        # new_image[:,:,3] = 1
        new_image = new_image*255
        return new_image.astype('uint8')

    k1 = k2 = 0.001
    background = [0.25, 0.25, 0.25]
    beta_DAPI = [9.147, 6.9215, 1.0]
    beta_eosin = [0.1, 15.8, 0.3]
    # dapi_image = dapi_image[:,:,0] + dapi_image[:,:,1]
    # eosin_image = eosin_image[:,:,0] + eosin_image[:,:,1]
    return createVirtualHE(dapi_image, eosin_image, k1, k2, background, beta_DAPI, beta_eosin)
    

def group_technical_effects(X, cluster_sign, batch_sign):
    '''Ref: https://doi.org/10.1038/s43588-025-00824-7'''

    assert X.shape[0] == np.array(cluster_sign).shape[0] == np.array(batch_sign).shape[0]

    ct_list = np.unique(cluster_sign)
    batch_list = np.unique(batch_sign)
    var_g = 0
    for celltype in ct_list:
        ct_selection = cluster_sign == celltype
        X_ct, batch_ct = X[ct_selection], batch_sign[ct_selection]
        var_ct = X_ct.var(0, ddof=0)*X_ct.shape[0]
        var_ctb = 0
        for batch in batch_list:
            batch_selection = batch_ct == batch
            X_ctb = X_ct[batch_selection]
            var_ctb = var_ctb + X_ctb.var(0, ddof=0)*X_ctb.shape[0]
        var_g = var_g + (var_ct - var_ctb)/X_ct.shape[0]
        
    return var_g


def HED_simulated_artifact(img, alpha_var=0.2, beta_var=0.2):
    """
    Simulates stain-related artifacts in H&E images following the method described in:

    Tellez D., Balkenhol M., Otte-Höller I., et al.
    "Whole-slide mitosis detection in H&E breast histology using PHH3 as a reference
    to train distilled stain-invariant convolutional networks."
    IEEE Transactions on Medical Imaging, 2018, 37(9): 2126-2136.
    """
    
    v_H = np.array([0.650, 0.704, 0.286])
    v_E = np.array([0.072, 0.990, 0.105])
    v_H = v_H / np.linalg.norm(v_H)
    v_E = v_E / np.linalg.norm(v_E)
    v_bg = np.cross(v_H, v_E)
    v_bg = v_bg / np.linalg.norm(v_bg)
    M = np.stack([v_H, v_E, v_bg], axis=1) 

    alpha = np.array([
        np.random.uniform(1 - alpha_var, 1 + alpha_var),  # H
        np.random.uniform(1 - alpha_var, 1 + alpha_var),  # E
        1.0  # D/background no disturb
    ])
    beta = np.array([
        np.random.uniform(-beta_var, beta_var),  # H
        np.random.uniform(-beta_var, beta_var),  # E
        0.0  # D/background no disturb
    ])

    '''
    For reproducibility, the randomly sampled values in our paper were:
        alpha = np.array([1.18004869, 0.82641303, 1.0])
        beta = np.array([-0.17289437, -0.14233893, 0])
    '''

    img_flatted = img.reshape(-1, 3)
    image_hed = -np.log(img_flatted.astype(np.float32) / 255.0 + 1e-6) @ np.linalg.inv(M)
    image_hed_prime = image_hed * alpha + beta
    image_rgb_prime = np.exp(-image_hed_prime @ M)
    
    img_simulate = (image_rgb_prime * 255.0).reshape(img.shape[0], img.shape[1], 3)
    img_simulate = np.clip(img_simulate, 0, 255).astype(np.uint8)

    return img_simulate

def detect_periodic_components(emb, axis, threshold=0.08, bin_width=1, max_freqs=3, verbose=False):
    """
    检测嵌入向量中沿指定轴的周期性伪影成分
    返回:
        artifact_dims: numpy数组
        peak_freqs: numpy数组, shape=(n_feats, max_freqs)
            每个维度对应的前 max_freqs 个峰值频率
    """
    # ========== 第一步：数据分箱（Binning） ==========
    min_counts_per_bin = 100  # 每个bin的最小样本数要求（未使用）
    x0 = np.min(axis)
    x1 = np.max(axis)
    bins = np.arange(x0, x1 + bin_width, bin_width)
    n_bins = len(bins)
    fs_bin = 1.0 / bin_width

    binned_vals = np.full((n_bins, emb.shape[1]), np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i, xi in enumerate(bins):
        mask = (axis >= xi) & (axis < xi + bin_width)
        counts[i] = mask.sum()
        if counts[i] >= 1:
            binned_vals[i, :] = np.nanmean(emb[mask, :], axis=0)

    # ========== 第二步：缺失值插值 ==========
    valid_mask = ~np.isnan(binned_vals).any(axis=1)  # 未使用
    for j in range(binned_vals.shape[1]):
        col = binned_vals[:, j]
        nans = np.isnan(col)
        if nans.any():
            idx = np.arange(len(col))
            good = ~nans
            if good.sum() >= 2:
                col[nans] = np.interp(idx[nans], idx[good], col[good])
            else:
                binned_vals[:, j] = np.nan

    # ========== 第三步：周期图分析 ==========
    n_feats = emb.shape[1]
    peak_power_ratio = np.zeros(n_feats)
    peak_freqs = np.zeros((n_feats, max_freqs), dtype=float)

    for j in range(n_feats):
        col = binned_vals[:, j]
        if np.isnan(col).any():
            peak_power_ratio[j] = 0.0
            peak_freqs[j, :] = 0.0
            continue

        f, Pxx = periodogram(col, fs=fs_bin, scaling='spectrum', window='hann', detrend='linear')
        if len(f) <= 1:
            peak_power_ratio[j] = 0.0
            peak_freqs[j, :] = 0.0
            continue

        # 排除 DC 分量
        f1 = f[1:]
        P1 = Pxx[1:]
        if len(P1) == 0:
            peak_power_ratio[j] = 0.0
            peak_freqs[j, :] = 0.0
            continue

        total_power = np.sum(P1) + 1e-10
        idx_max = np.argmax(P1)
        peak_power_ratio[j] = P1[idx_max] / total_power

        # 取前 max_freqs 个峰值频率
        top_idx = np.argsort(P1)[::-1][:max_freqs]
        freqs = f1[top_idx]
        # 填入（不足 max_freqs 时用 0 填充）
        peak_freqs[j, :len(freqs)] = freqs

    # ========== 第四步：识别伪影维度 ==========
    artifact_dims = np.where(peak_power_ratio > threshold)[0]
    if verbose:
        print(f"Detected {len(artifact_dims)} artifact dims (threshold={threshold})")
    return artifact_dims, peak_freqs


def remove_sinusoidal_component_single(emb, axis, freqs):
    X = []
    for f0 in freqs:
        X.append(np.sin(2*np.pi * f0 * axis))
        X.append(np.cos(2*np.pi * f0 * axis))
    X = np.column_stack(X)
    reg = LinearRegression(fit_intercept=False).fit(X, emb)
    pred = reg.predict(X)
    return emb - pred


def remove_sinusoidal_component(emb, axis, artifact_dims=None, min_peak_freq=0, max_peak_freq=999, verbose=False, **kwargs):
    """
    去除嵌入向量中的周期性伪影成分
    """
    if artifact_dims is None:
        artifact_dims, peak_freqs = detect_periodic_components(emb, axis, verbose=verbose, **kwargs)

    emb_clean = emb.copy()
    iterator = artifact_dims
    if verbose:
        iterator = tqdm(artifact_dims)
    for j in iterator:
        freqs = peak_freqs[j]
        # 过滤掉0值（填充值）和超出范围的频率
        freqs = freqs[(freqs > min_peak_freq) & (freqs < max_peak_freq) & (freqs > 0)]
        if len(freqs) == 0:
            continue
        emb_clean[:, j] = remove_sinusoidal_component_single(emb[:, j], axis, freqs)
    return emb_clean


def generate_hex_spot(spatial, x_interval=100, diameter=55, all_in=False):
    x, y = spatial[:, 0], spatial[:, 1]
    spatial = np.vstack([x, y]).T
    
    y_interval = x_interval*np.sqrt(3)
    
    x_start, x_end = 0, x.max()           
    y_start, y_end = 0, y.max() 
    spot_x1 = np.arange(x_start, x_end + x_interval, x_interval)
    spot_y1 = np.arange(y_start, y_end + y_interval, y_interval)
    spot_x1, spot_y1 = np.meshgrid(spot_x1, spot_y1)
    spot_x1 = spot_x1.reshape(-1)
    spot_y1 = spot_y1.reshape(-1)

    x_start, x_end = x_interval/2, x.max()
    y_start, y_end = y_interval/2, y.max()  
    spot_x2 = np.arange(x_start, x_end + x_interval, x_interval)
    spot_y2 = np.arange(y_start, y_end + y_interval, y_interval)
    spot_x2, spot_y2 = np.meshgrid(spot_x2, spot_y2)
    spot_x2 = spot_x2.reshape(-1)
    spot_y2 = spot_y2.reshape(-1)

    spot1 = np.vstack([spot_x1, spot_y1]).T
    spot2 = np.vstack([spot_x2, spot_y2]).T
    spot_spatial = np.vstack([spot1, spot2])
    
    tree = KDTree(spot_spatial)
    indices_list = tree.query_radius(spatial, r=diameter/2.0)
    rows = []
    cols = []
    for cell_idx, spot_indices in enumerate(indices_list):
        if len(spot_indices) == 0:
            continue
        for spot_idx in spot_indices:
            rows.append(spot_idx)   # spot
            cols.append(cell_idx)   # cell
    values = np.ones(len(rows))
    agg_mtx = sp.coo_matrix((values, (rows, cols)), shape=(spot_spatial.shape[0], spatial.shape[0])).tocsr()
    spot_selection = np.where((agg_mtx.sum(1) != 0).A.squeeze())[0]
    agg_mtx = agg_mtx[spot_selection, :]
    spot_spatial = spot_spatial[spot_selection]

    return spot_spatial, agg_mtx


def build_spot_abundance_matrix(adata, domain_names, prefix='Abundence_'):
    abundance_cols = [f'{prefix}{domain_name}' for domain_name in domain_names]
    missing_cols = [col for col in abundance_cols if col not in adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f'Missing deconvolution columns in spot data: {missing_cols}')
    return adata.obs[abundance_cols].to_numpy(dtype=np.float32)


def mean_annotation_embedding(abundance, anno_emb):
    abundance = abundance / abundance.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return abundance @ anno_emb.weight