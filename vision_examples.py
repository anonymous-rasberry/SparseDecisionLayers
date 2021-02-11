import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# Personal preference here to default to grad not enabled; 
# explicitly enable grad when necessary for memory reasons
ch.manual_seed(0)
ch.set_grad_enabled(False)

from robustness.datasets import ImageNet, Places365
from robustness.model_utils import make_and_restore_model
import cox
from cox import store

from elasticnet import glm_saga, IndexedTensorDataset
from torch.utils.data import TensorDataset, DataLoader

import os
import math
import time
from argparse import ArgumentParser

def get_deep_features(dataset, loader, model_root, arch, device='cuda'): 
    rng = np.random.RandomState(random_seed)

    model, _ = make_and_restore_model(arch=arch, 
             dataset=dataset,
             resume_path=model_root
        )
    model.eval()
    model = ch.nn.DataParallel(model.to(device))

    latents, labels = [], []
    for _, (X,y) in tqdm(enumerate(loader), total=len(loader)): 
        (op, reps), _ = model(X.to(device), with_latent=True)
        latents.append(reps.cpu())
        labels.append(y)
    return ch.cat(latents), ch.cat(labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/tmp/path_to_dataset')
    parser.add_argument('--dataset-name', type=str, default='places_small')
    parser.add_argument('--out-path', type=str, default='./glm_output')
    parser.add_argument('--model-root', type=str, default='/tmp/model_checkpoint.ckpt')
    parser.add_argument('--model-arch', type=str, default='resnet50')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--feat-batch-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--lam-factor', type=float, default=0.001)
    parser.add_argument('--group', type=bool, default=False)
    args = parser.parse_args()

    res_dir = os.path.join(args.out_path)
    if not os.path.exists(res_dir): 
        os.makedirs(res_dir)

    out_store = store.Store(res_dir)
    args_dict = args.as_dict() if isinstance(args, cox.utils.Parameters) else vars(args)

    
    print("Initializing dataset...")
    ds = Places365(os.path.join(args.dataset_path, args.dataset_name))
    _, test_loader = ds.make_loaders(batch_size=args.batch_size, workers=args.num_workers)
    NUM_CLASSES, ds.num_classes = 10, 10


    train_loader, test_loader = ds.make_loaders(args.num_workers, 
                                                args.feat_batch_size, 
                                                data_aug=False, 
                                                shuffle_train=False, 
                                                shuffle_val=False)

    print("Computing deep features...")
    X_tr, y_tr = get_deep_features(ds, train_loader, args.model_root, args.arch)
    X_te, y_te = get_deep_features(ds, test_loader, args.model_root, args.arch)

    print(f"  + Training data size: {X_tr.size()}")
    print(f"  + Test data size: {X_te.size()}")


    print("Standardizing data...")
    mu, std = X_tr.mean(0), X_tr.std(0)

    X_tr = (X_tr - mu)/std
    X_te = (X_te - mu)/std

    print("Creating validation set and tensor loaders...")
    val_sz = math.floor(X_tr.size(0)*args.val_frac)
    indices = ch.randperm(X_tr.size(0))
    X_val, X_tr = X_tr[indices[:val_sz]], X_tr[indices[val_sz:]]
    y_val, y_tr = y_tr[indices[:val_sz]], y_tr[indices[val_sz:]]
    ds_tr = IndexedTensorDataset(X_tr, y_tr)
    ds_val = TensorDataset(X_val, y_val)
    ds_te = TensorDataset(X_te, y_te)

    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    ld_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    Nfeatures = X_tr.size(1)

    args_dict['Nfeatures'] = Nfeatures
    args_dict['feature_indices'] = idx_sub
    schema = store.schema_from_dict(args_dict)
    out_store.add_table("metadata", schema)
    out_store["metadata"].append_row(args_dict)

    print("Initializing linear model...")
    linear = nn.Linear(Nfeatures, NUM_CLASSES).cuda()
    weight = linear.weight
    bias = linear.bias

    for p in [weight,bias]: 
        p.data.zero_()

    print("Calculating the regularization path")
    params = glm_saga(linear, ld_tr, args.lr, args.max_epochs, args.alpha, 
        n_classes=NUM_CLASSES, checkpoint=f"{res_dir}/params", 
        verbose=args.verbose, tol=args.tol, group=args.group, epsilon=args.lam_factor, 
        table_device='cpu', val_loader=ld_val, test_loader=ld_te)

   