import argparse
import os
from tqdm import tqdm
import numpy as np
import json
import time

import torch as ch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

    
def imagenet_repr_dataset(location, mode): 
    if not os.path.exists(os.path.join(location, f"{mode}_0_X.npy")): 
        raise ValueError(f"The provided location {location} does not contain any imagenet representation files")

    chunk_idx = 0
    ds_list = []
    start_time = time.time()
    while os.path.exists(os.path.join(location, f"{mode}_{chunk_idx}_X.npy")): 
        X = ch.from_numpy(np.load(os.path.join(location, f"{mode}_{chunk_idx}_X.npy"))).float()
        y = ch.from_numpy(np.load(os.path.join(location, f"{mode}_{chunk_idx}_y.npy"))).long()
        ds_list.append(TensorDataset(X,y))
        chunk_idx += 1

    print(f"==> loaded {chunk_idx} files of representations ({time.time()-start_time} seconds)")
    return ConcatDataset(ds_list)

def calculate_metadata(loader, num_classes=None): 
    # Calculate number of classes if not given
    if num_classes is None: 
        num_classes = 1
        for batch in loader:
            y = batch[1]
            print(y)
            num_classes = max(num_classes, y.max().item()+1)
    eye = ch.eye(num_classes)

    X_bar = 0
    y_bar = 0
    y_max = 0
    n = 0    
    # calculate means and maximum
    print("Calculating means")
    for X,y in tqdm(loader):
        X_bar += X.sum(0)
        y_bar += eye[y].sum(0)
        y_max = max(y_max, y.max())
        n += y.size(0)
    X_bar = X_bar.float()/n
    y_bar = y_bar.float()/n

    # calculate std
    X_std = 0
    y_std = 0
    print("Calculating standard deviations")
    for X,y in tqdm(loader): 
        X_std += ((X - X_bar)**2).sum(0)
        y_std += ((eye[y] - y_bar)**2).sum(0)
    X_std = ch.sqrt(X_std.float()/n)
    y_std = ch.sqrt(y_std.float()/n)

    # calculate maximum regularization
    inner_products = 0
    print("Calculating maximum lambda")
    for X,y in tqdm(loader): 
        y_map = (eye[y] - y_bar)/y_std

        inner_products += X.t().mm(y_map)*y_std

    inner_products_group = inner_products.norm(p=2,dim=1)

    return {
        "X": {
            "mean": X_bar, 
            "std": X_std, 
            "num_features": X.size()[1:], 
            "num_examples": n
        }, 
        "y": {
            "mean": y_bar, 
            "std": y_std,
            "num_classes": y_max+1
        },
        "max_reg": {
            "group": inner_products_group.abs().max().item()/n, 
            "nongrouped": inner_products.abs().max().item()/n
        }
    }

# Main function here creates and saves imagenet features
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Create Imagenet representations')
    parser.add_argument('--dataset-path', type=str, default='/tmp/path_to_imagenet')
    parser.add_argument('--model-root', type=str, default='/tmp/model_checkpoint.ckpt')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--out-path', default='/tmp/path_to_imagenet_features')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--chunk-threshold', default=20000, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--max-chunks', default=-1, type=int)

    args = parser.parse_args()

    # Personal preference here to default to grad not enabled; 
    # explicitly enable grad when necessary for memory reasons
    ch.manual_seed(0)
    ch.set_grad_enabled(False)

    print("Initializing dataset and loader...")
    ds_imagenet = ImageNet(args.dataset_path)
    train_loader, test_loader = ds_imagenet.make_loaders(args.num_workers, args.batch_size, 
                                                        data_aug=False, shuffle_train=False, shuffle_val=False)

    print("Loading model...")
    model, _ = make_and_restore_model( 
        arch=args.arch, 
        dataset=ds_imagenet,
        resume_path=args.model_root
    )
    model.eval()
    model = ch.nn.DataParallel(model.to(args.device))

    out_dir = args.out_path
    if not os.path.exists(out_dir):
        print(f"Making directory {out_dir}")
        os.makedirs(out_dir)

    for mode,loader in zip(['train', 'test'], [train_loader, test_loader]): 
        print(f"Creating {mode} features in {out_dir}")
        all_latents, all_labels = [], []

        chunk_id, n = 0, 0
        for X,y in tqdm(loader): 
            (_,latents), _ = model(X.to(args.device), with_latent=True)
            all_latents.append(latents.cpu())    
            all_labels.append(y)
            if n == 0 and chunk_id == 0:
                print("Latents shape", latents.shape)
            n += X.size(0)            

            if n > args.chunk_threshold: 
                np.save(os.path.join(out_dir,f'{mode}_{chunk_id}_X.npy'), ch.cat(all_latents).numpy())
                np.save(os.path.join(out_dir,f'{mode}_{chunk_id}_y.npy'), ch.cat(all_labels).numpy())
                all_latents = []
                all_labels = []
                n = 0
                chunk_id += 1
                if args.max_chunks > 0 and chunk_id >= args.max_chunks: 
                    print(f"Hit max-chunks {max_chunks}, stopping early just for testing purposes")
                    break

        if n > 0:
            # save last chunk 
            np.save(os.path.join(out_dir,f'{mode}_{chunk_id}_X.npy'), ch.cat(all_latents).numpy())
            np.save(os.path.join(out_dir,f'{mode}_{chunk_id}_y.npy'), ch.cat(all_labels).numpy())


    print("Loading representation dataloaders...")
    repr_train_dataset = imagenet_repr_dataset(out_dir, "train")
    repr_train_loader = DataLoader(repr_train_dataset, batch_size=args.batch_size, shuffle=False)
    repr_test_dataset = imagenet_repr_dataset(out_dir, "test")
    repr_test_loader = DataLoader(repr_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Calculating metadata...")
    metadata = calculate_metadata(repr_train_loader, num_classes=1000)
    ch.save(metadata, os.path.join(out_dir,'metadata_train.pth'))
    metadata = calculate_metadata(repr_test_loader, num_classes=1000)
    ch.save(metadata, os.path.join(out_dir,'metadata_test.pth'))