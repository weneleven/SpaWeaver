import os
import torch
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import SpaWeaver as sw
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def train(rank, world_size, args, adata1, adata2, models):
    sw.utils.setup(rank, world_size)

    models = models.to(rank)
    model_1, mlp_1, reg_1, reg_2 = models
    models = torch.nn.parallel.DistributedDataParallel(models, device_ids=[rank])

    optimizer = sw.utils.create_optimizer(args.optimizer, models, args.lr, args.weight_decay)

    node_HE_fea1, node_HE_fea2 = torch.Tensor(adata1.obsm['he_sp']), torch.Tensor(adata2.obsm['he_sp'])
    X1_real, X2_real = torch.Tensor(adata1.X), torch.Tensor(adata2.X)
    dataset1 = TensorDataset(X1_real, node_HE_fea1)
    dataset2 = TensorDataset(X2_real, node_HE_fea2)
    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    sampler2 = DistributedSampler(dataset2, num_replicas=world_size, rank=rank)
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1, drop_last=True)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, sampler=sampler2, drop_last=True)

    rbf = sw.model.RBF().to(rank)
    MMD = sw.model.MMD_loss(kernel=rbf).to(rank)

    len1 = len(dataloader1)
    len2 = len(dataloader2)

    if rank == 0:
        print(f"====== Dataset Comprison: Data 1={len1} batches, Data 2={len2} batches ======")
        if len2 > len1:
            print(f"Data 2 data more，Automatically iterate over the Data 1 dataset...")

    
    epoch_iter = tqdm(range(args.epoch), desc="🧠 Training", disable=(rank != 0))
    for epoch in epoch_iter:
        sampler1.set_epoch(epoch)
        sampler2.set_epoch(epoch)

        if len1 >= len2:
            # If the dataset1 is larger, iterate over the dataset2 instead
            iterator = zip(dataloader1, itertools.cycle(dataloader2))
        else:
            # If the dataset2 is larger, iterate over the dataset1 instead.
            iterator = zip(itertools.cycle(dataloader1), dataloader2)

        for data1, data2 in iterator:
            X1_real_batch, node_HE_fea1_batch = data1
            X2_real_batch, node_HE_fea2_batch = data2
            X1_real_batch, node_HE_fea1_batch = X1_real_batch.to(rank), node_HE_fea1_batch.to(rank)
            X2_real_batch, node_HE_fea2_batch = X2_real_batch.to(rank), node_HE_fea2_batch.to(rank)

            he1_map, he2_map = model_1(mlp_1(node_HE_fea1_batch)), model_1(mlp_1(node_HE_fea2_batch))
            rec_omics1 = reg_1(he1_map)
            rec_omics2 = reg_2(he2_map)

            # loss
            mmd = MMD(he1_map, he2_map)
            loss1 = F.mse_loss(rec_omics1, X1_real_batch)
            loss2 = F.mse_loss(rec_omics2, X2_real_batch)
            loss = loss1 + loss2 + args.mmd_weight * mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    sw.utils.cleanup()

    if rank == 0:
        print("💾 Saving checkpoint...")
        model_1_save = model_1.module if hasattr(model_1, "module") else model_1
        mlp_1_save = mlp_1.module if hasattr(mlp_1, "module") else mlp_1
        reg_1_save = reg_1.module if hasattr(reg_1, "module") else reg_1
        reg_2_save = reg_2.module if hasattr(reg_2, "module") else reg_2

        torch.save({
            'model_1': model_1_save.state_dict(),
            'mlp_1': mlp_1_save.state_dict(),
            'reg_1': reg_1_save.state_dict(),
            'reg_2': reg_2_save.state_dict()
        }, f'{args.output_folder}model/{args.save_tag}/models.pt')
        print(f"✅ Checkpoint saved to {args.output_folder}model/{args.save_tag}/models.pt.")


def train_Multi_Omics(rank, world_size, args, adata1, adata2, models, anno1, anno2):
    sw.utils.setup(rank, world_size)

    models = models.to(rank)
    model_1, mlp_1, reg_1, reg_2, anno_emb = models
    models = torch.nn.parallel.DistributedDataParallel(models, device_ids=[rank])

    optimizer = sw.utils.create_optimizer(args.optimizer, models, args.lr, args.weight_decay)

    node_HE_fea1, node_HE_fea2 = torch.Tensor(adata1.obsm['he_sp']), torch.Tensor(adata2.obsm['he_sp'])
    X1_real, X2_real = torch.Tensor(adata1.X), torch.Tensor(adata2.X)
    anno1, anno2 = torch.LongTensor(anno1), torch.LongTensor(anno2)

    dataset1 = TensorDataset(X1_real, node_HE_fea1, anno1)
    dataset2 = TensorDataset(X2_real, node_HE_fea2, anno2)
    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank)
    sampler2 = DistributedSampler(dataset2, num_replicas=world_size, rank=rank)
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, sampler=sampler1, drop_last=False)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, sampler=sampler2, drop_last=False)

    rbf = sw.model.RBF().to(rank)
    MMD = sw.model.MMD_loss(kernel=rbf).to(rank)

    epoch_iter = tqdm(range(args.epoch), desc="🧠 Training", disable=(rank != 0))
    for epoch in epoch_iter:    
        sampler1.set_epoch(epoch)
        sampler2.set_epoch(epoch)

        for data1, data2 in zip(dataloader1, dataloader2):
            X1_real_batch, node_HE_fea1_batch, anno1_batch = data1
            X2_real_batch, node_HE_fea2_batch, anno2_batch = data2
            X1_real_batch, node_HE_fea1_batch, anno1_batch = X1_real_batch.to(rank), node_HE_fea1_batch.to(rank), anno1_batch.to(rank)
            X2_real_batch, node_HE_fea2_batch, anno2_batch = X2_real_batch.to(rank), node_HE_fea2_batch.to(rank), anno2_batch.to(rank)

            anno_emb1, anno_emb2 = F.dropout(anno_emb(anno1_batch), p=0.1), F.dropout(anno_emb(anno2_batch), p=0.1)
            he1_map, he2_map = model_1(mlp_1(node_HE_fea1_batch)), model_1(mlp_1(node_HE_fea2_batch))    
            emb1, emb2 = torch.concat([he1_map, anno_emb1], dim=1), torch.concat([he2_map, anno_emb2], dim=1)
            rec_omics1, rec_omics2 = reg_1(emb1), reg_2(emb2)

            loss1 = F.mse_loss(rec_omics1, X1_real_batch)
            loss2 = F.mse_loss(rec_omics2, X2_real_batch)
            mmd = MMD(he1_map, he2_map)
            loss = loss1 + loss2 + args.mmd_weight * mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    sw.utils.cleanup()

    if rank == 0:
        print("💾 Saving checkpoint...")
        model_1_save = model_1.module if hasattr(model_1, "module") else model_1
        mlp_1_save = mlp_1.module if hasattr(mlp_1, "module") else mlp_1
        reg_1_save = reg_1.module if hasattr(reg_1, "module") else reg_1
        reg_2_save = reg_2.module if hasattr(reg_2, "module") else reg_2
        anno_emb_save = anno_emb.module if hasattr(anno_emb, "module") else anno_emb
        torch.save({
            'model_1': model_1_save.state_dict(),
            'mlp_1': mlp_1_save.state_dict(),
            'reg_1': reg_1_save.state_dict(),
            'reg_2': reg_2_save.state_dict(),
            'anno_emb': anno_emb_save.state_dict(),
        }, f'{args.output_folder}model/{args.save_tag}/models.pt')


def train_Cross_Resolution(rank, world_size, args, adata1, adata2, models, agg_mtx2, anno1, anno2):
    sw.utils.setup(rank, world_size)

    models = models.to(rank)
    model_1, mlp_1, mlp_2, reg_1, reg_2, anno_emb = models
    models = torch.nn.parallel.DistributedDataParallel(models, device_ids=[rank])
    optimizer = sw.utils.create_optimizer(args.optimizer, models, args.lr, args.weight_decay)

    he1, he2 = torch.Tensor(adata1.obsm['sp_he']), torch.Tensor(adata2.obsm['sp_he'])
    panelA1, panelB2 = torch.Tensor(adata1.X), torch.Tensor(adata2.X)
    anno1, anno2 = torch.FloatTensor(anno1).to(rank), torch.LongTensor(anno2).to(rank)

    dataset2 = TensorDataset(torch.arange(agg_mtx2.shape[0]))
    batch_size2 = args.batch_size
    sampler2 = DistributedSampler(dataset2, num_replicas=world_size, rank=rank)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size2, sampler=sampler2, drop_last=False)
    he1_batch = he1.to(rank)
    panelA1_batch = panelA1.to(rank)

    rbf = sw.model.RBF().to(rank)
    MMD = sw.model.MMD_loss(kernel=rbf).to(rank)

    print('================================ Train ================================\n')
    for epoch in range(args.epoch):
        sampler2.set_epoch(epoch)
        batch_iter = tqdm(
            dataloader2,
            desc=f'🚀 Epoch {epoch + 1}/{args.epoch}',
            leave=False,
        )
        for spot_idx2 in batch_iter:
            cell_idx2 = agg_mtx2[spot_idx2[0].to(int)].tocoo().col
            agg_mtx2_batch = sw.utils.sparse_mx_to_torch_sparse_tensor(agg_mtx2[spot_idx2[0]][:, cell_idx2], device=rank)
            panelB2_batch, he2_batch = panelB2[cell_idx2].to(rank), he2[cell_idx2].to(rank)

            anno_emb1 = F.dropout(sw.utils.mean_annotation_embedding(anno1, anno_emb), p=0)
            anno_emb2 = F.dropout(anno_emb(anno2[cell_idx2]), p=0)
            he1_map, he2_map = model_1(mlp_1(he1_batch)), model_1(mlp_2(he2_batch))
            emb1, emb2 = torch.concat([he1_map, anno_emb1], dim=1), torch.concat([he2_map, anno_emb2], dim=1)
            rec_omics1 = reg_1(emb1)
            rec_omics2 = reg_2(emb2)

            mmd = MMD(he1_map, agg_mtx2_batch@he2_map)
            loss1 = F.mse_loss(rec_omics1, panelA1_batch)
            loss2 = F.mse_loss(rec_omics2, panelB2_batch)
            loss = loss1 + loss2 + args.mmd_weight * mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                batch_iter.set_postfix({
                    'loss1': f'{loss1.item():.4f}',
                    'loss2': f'{loss2.item():.4f}',
                    'mmd': f'{mmd.item():.4f}'
                })
    sw.utils.cleanup()

    if rank == 0:
        model_1_save = model_1.module if hasattr(model_1, "module") else model_1
        mlp_1_save = mlp_1.module if hasattr(mlp_1, "module") else mlp_1
        mlp_2_save = mlp_2.module if hasattr(mlp_2, "module") else mlp_2
        reg_1_save = reg_1.module if hasattr(reg_1, "module") else reg_1
        reg_2_save = reg_2.module if hasattr(reg_2, "module") else reg_2
        anno_emb_save = anno_emb.module if hasattr(anno_emb, "module") else anno_emb
        torch.save({
            'model_1': model_1_save.state_dict(),
            'mlp_1': mlp_1_save.state_dict(),
            'mlp_2': mlp_2_save.state_dict(),
            'reg_1': reg_1_save.state_dict(),
            'reg_2': reg_2_save.state_dict(),
            'anno_emb': anno_emb_save.state_dict(),
        },  f'{args.output_folder}model/{args.save_tag}/models.pt')