import torch as ch
import torch.nn as nn
# Personal preference here to default to grad not enabled; 
# explicitly enable grad when necessary for memory reasons
ch.manual_seed(0)
ch.set_grad_enabled(False)

from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model

from elasticnet import glm_saga, add_index_to_dataloader, NormalizedRepresentation
from imagenet_features import imagenet_repr_dataset

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import math
import time
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--feature-path', type=str, default='/tmp/path_to_imagenet_features')
    parser.add_argument('--dataset-name', type=str, default='imagenet')
    parser.add_argument('--out-path', type=str, default='./output')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--lr-decay-factor', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--lookbehind', type=int, default=3)
    parser.add_argument('--lam-factor', type=float, default=0.001)
    parser.add_argument('--group', type=bool, default=False)
    args = parser.parse_args()


    res_dir = args.out_path
    if not os.path.exists(res_dir): 
        os.makedirs(res_dir)

    start_time = time.time()

    # Load precomputed features
    repr_dir = args.feature_path

    
    # the encoder just performs normalization of the dataset
    print("Load precalculated metadata...")
    metadata = ch.load(os.path.join(repr_dir, 'metadata_train.pth'))
    NUM_FEATURES = metadata["X"]["num_features"][0]
    NUM_CLASSES = metadata["y"]["num_classes"].numpy()

    
    print(f"Initializing dataset from {repr_dir}...")
    train_ds = imagenet_repr_dataset(repr_dir, "train")
    test_ds = imagenet_repr_dataset(repr_dir, "test")
    
    print("Creating validation set and tensor loaders...")
    total_sz = len(train_ds)
    val_sz = math.floor(total_sz*args.val_frac)
    print(total_sz, val_sz)
    
    train_ds, val_ds = ch.utils.data.random_split(train_ds, 
                                                  [total_sz - val_sz, val_sz], 
                                                  generator=ch.Generator().manual_seed(0))
    
    train_loader = DataLoader(train_ds, 
                              num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, 
                              num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, 
                              num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    print("Initializing linear model...")
    linear = nn.Linear(NUM_FEATURES, NUM_CLASSES).cuda()
    weight = linear.weight
    bias = linear.bias
    
    print("Preparing normalized encoder and indexed dataloader")
    encoder = NormalizedRepresentation(train_loader, metadata=metadata, device=linear.weight.device)
    indexed_train_loader = add_index_to_dataloader(train_loader)
    indexed_val_loader = add_index_to_dataloader(val_loader)
    indexed_test_loader = add_index_to_dataloader(test_loader)
    
    
    for p in [weight,bias]: 
        p.data.zero_()

    CHECKPOINT = f'{res_dir}/checkpoint'
    if not os.path.exists(CHECKPOINT): 
        os.makedirs(CHECKPOINT, exist_ok=True)

    print("Calculating the regularization path")
    params = glm_saga(linear, indexed_train_loader, 
                      args.lr, args.max_epochs, args.alpha, 
                      n_classes=NUM_CLASSES, checkpoint=CHECKPOINT,
                      verbose=args.verbose, tol=args.tol, 
                      lookbehind=args.lookbehind, lr_decay_factor=args.lr_decay_factor,
                      group=args.group, epsilon=args.lam_factor, metadata=metadata,
                      encoder=encoder, val_loader=indexed_val_loader,
                      test_loader=indexed_test_loader)

    print(f"Total time: {time.time() - start_time}")
