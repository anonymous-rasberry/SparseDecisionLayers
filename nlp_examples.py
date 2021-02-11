import os, math, sys
from tqdm import tqdm
from time import time
import numpy as np
import torch as ch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

from transformers import AutoTokenizer, AutoConfig
from nlp_modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import RobertaForSequenceClassification
from nlp_dataset import SSTDataset, YelpDataset, JigsawDataset

from lime.lime_text import LimeTextExplainer

from elasticnet import glm_saga, IndexedTensorDataset
from nlp_evaluate import evaluate
from argparse import ArgumentParser

MODEL_DICT = {
    'sst': 'barissayil/bert-sentiment-analysis-sst',
    'jigsaw-toxic': 'unitary/toxic-bert', 
    'jigsaw-severe_toxic': 'unitary/toxic-bert', 
    'jigsaw-obscene': 'unitary/toxic-bert', 
    'jigsaw-threat': 'unitary/toxic-bert', 
    'jigsaw-insult': 'unitary/toxic-bert', 
    'jigsaw-identity_hate': 'unitary/toxic-bert', 
    'jigsaw-alt-toxic': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-severe_toxic': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-obscene': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-threat': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-insult': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-identity_hate': 'unitary/unbiased-toxic-roberta'
}


def make_bert_features(loader, model, device, pooled_output=False): 
    X = []
    y = []
    for i, (input_ids, attention_mask, labels) in enumerate(tqdm(iterable=loader, desc="Generating features")):
        mask = labels != -1
        input_ids, attention_mask, labels = [t[mask] for t in (input_ids, attention_mask, labels)]
        if hasattr(model, "roberta"): 
            output = model.roberta(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            output = output[0]

            # do RobertA classification head minus last out_proj classifier
            # https://huggingface.co/transformers/_modules/transformers/models/roberta/modeling_roberta.html
            output = output[:,0,:]
            output = model.classifier.dropout(output)
            output = model.classifier.dense(output)
            output = ch.tanh(output)
            output = model.classifier.dropout(output)
        else: 
            output = model.bert(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            # Extra layer according to https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification
            if pooled_output: 
                output = output[1]
            else: 
                output = output[0][:,0]
        X.append(output.cpu())
        y.append(labels.clone())
    return ch.cat(X,dim=0), ch.cat(y,dim=0).long()


def balance_reps(norm_rep, rep_label): 
    n0 = rep_label.sum().item()
    n = rep_label.size(0)
    I_pos = rep_label == 1
    rep_pos, label_pos = norm_rep[I_pos], rep_label[I_pos]
    ch.manual_seed(0)
    I = ch.randperm(n - n0)[:n0]
    rep_neg, label_neg = norm_rep[~I_pos][I], rep_label[~I_pos][I]

    return ch.cat([rep_pos, rep_neg],dim=0), ch.cat([label_pos,label_neg],dim=0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-name-or-path', type=str, default=None)
    parser.add_argument('--out-path', type=str, default='./glm_output')
    parser.add_argument('--cache', type=str, default='./cache')
    parser.add_argument('--dataset', type=str, default='yelp-polarity-alt')
    parser.add_argument('--maxlen_train', type=int, default=256, 
                        help='Maximum number of tokens in the input sequence during training.')
    parser.add_argument('--maxlen_val', type=int, default=256, 
                        help='Maximum number of tokens in the input sequence during evaluation.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--epsilon-list', type=list, default=[3])
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr-decay-factor', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--verbose', type=int, default=50)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--lam-factor', type=float, default=1e-3)
    parser.add_argument('--lookbehind', type=int, default=5)
    parser.add_argument('--k', type=int, default=100)
    args = parser.parse_args()

    print(" ")
    print(args)

    model_name_or_path = args.model_name_or_path
    if model_name_or_path is None:
        model_name_or_path = MODEL_DICT[args.dataset]

    print(f'Model name: {model_name_or_path}')
    #Configuration for the desired transformer model
    config = AutoConfig.from_pretrained(model_name_or_path)

    print('Please wait while the analyser is being prepared.')

    #Create the model with the desired transformer model
    if config.model_type == 'bert':
        if model_name_or_path == 'barissayil/bert-sentiment-analysis-sst': 
            model = BertForSentimentClassification.from_pretrained(model_name_or_path)
            pooled_output = False
        else: 
            model = BertForSequenceClassification.from_pretrained(model_name_or_path)
            pooled_output = True
    elif config.model_type == 'roberta': 
        model = RobertaForSequenceClassification.from_pretrained(model_name_or_path)
        pooled_output = False

    elif config.model_type == 'albert':
        model = AlbertForSentimentClassification.from_pretrained(model_name_or_path)
    elif config.model_type == 'distilbert':
        model = DistilBertForSentimentClassification.from_pretrained(model_name_or_path)
    else:
        raise ValueError('This transformer model is not supported yet.')

    device = ch.device("cuda" if ch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()

    #Initialize the tokenizer for the desired transformer model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading dataset {args.dataset}")
    if args.dataset == 'sst':
        train_set = SSTDataset(filename='data/sst/train.tsv', maxlen=args.maxlen_train, tokenizer=tokenizer)
        val_set = SSTDataset(filename='data/sst/dev.tsv', maxlen=args.maxlen_val, tokenizer=tokenizer)
    elif args.dataset.startswith('yelp-polarity'):
        train_set = YelpDataset(filename='data/yelp_review_polarity_csv/train.csv', maxlen=args.maxlen_train, tokenizer=tokenizer)
        val_set = YelpDataset(filename='data/yelp_review_polarity_csv/test.csv', maxlen=args.maxlen_val, tokenizer=tokenizer)
    elif args.dataset == 'yelp-review':
        train_set = YelpDataset(filename='data/yelp_review_full_csv/train.csv', maxlen=args.maxlen_train, tokenizer=tokenizer)
        val_set = YelpDataset(filename='data/yelp_review_full_csv/test.csv', maxlen=args.maxlen_val, tokenizer=tokenizer)
    elif 'jigsaw' in args.dataset: 
        if 'alt' in args.dataset: 
            toxicity_type = args.dataset[11:]
        else: 
            toxicity_type = args.dataset[7:]
        train_set = JigsawDataset(filename='data/jigsaw_data/train.csv', maxlen=args.maxlen_train, tokenizer=tokenizer, label=toxicity_type)
        val_set = JigsawDataset(filename='data/jigsaw_data/test.csv', maxlen=args.maxlen_train, tokenizer=tokenizer, label=toxicity_type)
    else:
        raise ValueError(f"Dataset {args.dataset} is not implemented yet")

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_workers)


    with ch.no_grad(): 
        criterion = nn.BCEWithLogitsLoss()

        cache_dir = f"{args.cache}/{args.dataset}_{model_name_or_path}"
        if not os.path.exists(cache_dir):
            print(f"Making directory {cache_dir}")
            os.makedirs(cache_dir)

        cached_tr = os.path.join(cache_dir, "train_features.pth")
        if os.path.exists(cached_tr): 
            (X_tr, y_tr) = ch.load(cached_tr)
        else: 
            # generate features
            print("Making bert features")
            X_tr, y_tr = make_bert_features(train_loader, model, device, pooled_output=pooled_output)
            ch.save((X_tr, y_tr), cached_tr)

        cached_te = os.path.join(cache_dir, "test_features.pth")
        if os.path.exists(cached_te): 
            print("loading cached features")
            (X_te,y_te) = ch.load(cached_te)
        else:
            # generate features
            print("Making bert features")
            X_te, y_te = make_bert_features(val_loader, model, device, pooled_output=pooled_output)
            ch.save((X_te,y_te), cached_te)

        NUM_CLASSES = int(y_tr.max().item()+1)

        print(f"  + Training data size: {X_tr.size()}")

        print("Standardizing data...")
        mu, std = X_tr.mean(0), ch.clamp(X_tr.std(0), min=1e-5)

        X_tr = (X_tr - mu)/std
        X_te = (X_te - mu)/std

        print("Creating validation set and tensor loaders...")
        val_sz = math.floor(X_tr.size(0)*args.val_frac)
        indices = ch.randperm(X_tr.size(0))
        X_val, X_tr = X_tr[indices[:val_sz]], X_tr[indices[val_sz:]]
        y_val, y_tr = y_tr[indices[:val_sz]], y_tr[indices[val_sz:]]

        X_val, y_val = balance_reps(X_val, y_val)
        X_te, y_te = balance_reps(X_te, y_te)

        print(y_val.float().mean())

        ds_tr = IndexedTensorDataset(X_tr, y_tr)
        ds_val = TensorDataset(X_val, y_val)
        ds_te = TensorDataset(X_te, y_te)

        ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
        ld_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
        ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

        print("Initializing linear model...")
        linear = nn.Linear(X_tr.size(1), NUM_CLASSES).cuda()
        weight = linear.weight
        bias = linear.bias

        for p in [weight,bias]: 
            p.data.zero_()

        #ch.save(kwargs, )
        out_dir = f"{args.out_path}/{args.dataset}_{model_name_or_path}"
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_dir):
            print(f"Making directory {out_dir}")
            os.makedirs(out_dir)

        print(f"Calculating the regularization path in {out_dir}") 
        start_time = time()
        params = glm_saga(linear, ld_tr, args.lr, args.max_epochs, args.alpha, 
                          n_classes=NUM_CLASSES, checkpoint=out_dir, verbose=args.verbose, 
                          tol=args.tol, group=False, epsilon=args.lam_factor,
                          val_loader=ld_val, test_loader=ld_te, lr_decay_factor=args.lr_decay_factor, 
                          lookbehind=args.lookbehind, k=args.k)
        print(f"Total time: {time() - start_time}")


