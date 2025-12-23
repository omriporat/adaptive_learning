#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:26:20 2025

@author: itayta
"""

import argparse
import json
import os
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE

from plm_base import plm_init
from dataset import EpiNNetDataset, EpiNNetActivityTrainTest
from models import EpiNNet, plmTrunkModel
from config import load_config

parser = argparse.ArgumentParser(
    description="Train and evaluate EpiNNet or PLM models."
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the YAML configuration file (default: config.yaml)",
)
parser.add_argument("--root_path", type=str, help="Root path for the project")
parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file")
parser.add_argument("--save_path", type=str, help="Path to save outputs")
parser.add_argument("--weights_path", type=str, help="Path to model weights")
parser.add_argument(
    "--model_type", type=str, help="Type of model to use (e.g., 'plm', 'epinnet')"
)
parser.add_argument(
    "--train_type", type=str, help="Type of training to use (e.g., 'direct_mlp')"
)
parser.add_argument(
    "--nmuts_column", type=str, help="Column name for number of mutations"
)
parser.add_argument("--sequence_column_name", type=str, help="Column name for sequence")
parser.add_argument("--activity_column_name", type=str, help="Column name for activity")
parser.add_argument(
    "--first_column_name", type=str, help="First column name for encoding"
)
parser.add_argument(
    "--last_column_name", type=str, help="Last column name for encoding"
)
parser.add_argument("--plm_name", type=str, help="PLM model name")
parser.add_argument("--ref_seq", type=str, help="Reference sequence")
parser.add_argument(
    "--train_indices", type=int, nargs="+", help="List of train indices"
)
parser.add_argument("--test_indices", type=int, nargs="+", help="List of test indices")
parser.add_argument(
    "--pos_to_use", default=None, type=int, nargs="+", help="List of positions to use"
)
parser.add_argument(
    "--load_weights",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to load weights (True/False)",
)
parser.add_argument(
    "--train",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to train (True/False)",
)
parser.add_argument(
    "--train_indices_rev",
    type=lambda x: (str(x).lower() == "true"),
    help="Reverse train indices (True/False)",
)
parser.add_argument(
    "--test_indices_rev",
    type=lambda x: (str(x).lower() == "true"),
    help="Reverse test indices (True/False)",
)
parser.add_argument(
    "--evaluate_train",
    type=lambda x: (str(x).lower() == "true"),
    help="Evaluate on train set (True/False)",
)
parser.add_argument(
    "--evaluate_test",
    type=lambda x: (str(x).lower() == "true"),
    help="Evaluate on test set (True/False)",
)
parser.add_argument(
    "--train_drop_tokens",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to drop tokens during training (True/False)",
)
parser.add_argument(
    "--inference_drop_tokens",
    type=lambda x: (str(x).lower() == "true"),
    help="Whether to drop tokens during inference (True/False)",
)
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--iterations", type=int, help="Number of iterations")

parser.add_argument("--bootstrap_index", type=int, help="Bootstrap index to use", default=1)

# After loading config, overwrite config values with any provided CLI args

args = parser.parse_args()

# config_path = "epinnet_config.yaml"
# Load configuration from YAML file
config = load_config()

# Overwrite config values with any provided CLI args (if not None), dynamically
for key, val in vars(args).items():
    if key == "config":
        continue  # skip config path itself

    if val is not None:
        config[key] = val


if "root_path" in config:
    for k in list(config.keys()):
        if k.endswith("_path") and k != "root_path" and config[k] is not None:
            # Only update if not already an absolute path
            if not os.path.isabs(str(config[k])):
                config[k] = os.path.join(config["root_path"], str(config[k]))


plm_init(config["root_path"])

CE_IGNORE_INDEX = -66


def pairwise_cosine(X):
    X = F.normalize(X, dim=-1)
    similarity = torch.matmul(X, X.t())  # [N, N]
    distance = 1 - similarity
    return distance


def online_mine_triplets(labels):
    triplets = []

    for i, anchor_label in enumerate(labels):
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

        for pos_idx in positive_indices:
            if pos_idx == i:
                continue
            for neg_idx in negative_indices:
                triplets.append((i, pos_idx.item(), neg_idx.item()))

    return triplets


def get_indices(
    sequence_df, nmuts, nmuts_column="num_of_muts", rev=False, verbose=False
):
    indices = np.repeat(False, sequence_df.shape[0])

    if type(nmuts) == int:
        nmuts = [nmuts]

    for nm in nmuts:
        indices = indices | (sequence_df[nmuts_column] == nm).to_numpy()
        if verbose:
            print("Indices included: %d" % sum(indices))

    if rev:
        indices = ~indices

    return np.where(indices)[0].tolist()


def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]

    one_hot_encoding = torch.from_numpy(
        pd.get_dummies(sdf[sdf.columns[si:ei]]).to_numpy()
    ).to(torch.int64)

    return one_hot_encoding


def select_design_pos(seq):
    return [seq[21], seq[23], seq[24]]


# epinnet = SeqMLP("plm_embedding", 3, select_design_pos, plm_name="esm2_t12_35M_UR50D")

# epinnet.encode("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN")

def train_plm_triplet_model(
    plm_name: str,
    save_path: str,
    train_test_dataset,
    pos_to_use=None,
    batch_size=32,
    iterations=20000,
    margin=1.0,
    lr=4e-6,
    weight_decay=0.1,
    encoding_identifier=None,
    opmode="mean",
    hidden_layers=[1024],
    activation="sigmoid",
    train_type="triplet",
    layer_norm=False,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
    model=None,
    save_models=True
):
    print("\n[DEBUG] Training parameters:")
    print(f"\tplm_name: {plm_name}")
    print(f"\tsave_path: {save_path}")
    print("\ttrain_dataset size: %d" % len(train_test_dataset.train_dataset))
    print(f"\tpos_to_use: {pos_to_use}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\titerations: {iterations}")
    print(f"\tmargin: {margin}")
    print(f"\tlr: {lr}")
    print(f"\tweight_decay: {weight_decay}")
    print(f"\tencoding_identifier: {encoding_identifier}")
    print(f"\topmode: {opmode}")
    print(f"\thidden_layers: {hidden_layers}")
    print(f"\tactivation: {activation}")

    print(f"\tlayer_norm: {layer_norm}")
    print(f"\tactivation_on_last_layer: {activation_on_last_layer}")
    print(f"\tdevice: {device}")
    print(f"\tmodel: {type(model)}\n")
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoints_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # In the future we might have other use cases,
    emb_only = train_type == "triplet"

    if model is None:
        model = plmTrunkModel(
            plm_name=plm_name,
            opmode=opmode,
            emb_only=emb_only,
            specific_pos=pos_to_use,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device,
        ).to(device)
    else:
        model = model.to(device)

    if train_type == "direct_mlp":
        for layer in model.epinnet_trunk.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    if train_test_dataset is None:
        raise ValueError("train_test_dataset must be provided as an argument.")

    if pos_to_use is None:
        pos_to_use = [
            int(x[1:])
            for x in train_test_dataset.train_dataset.sequence_dataframe.columns[
                3:25
            ].tolist()
        ]

    print(f"Using positions: {pos_to_use}")
    train_loader = torch.utils.data.DataLoader(
        train_test_dataset, batch_size=batch_size, shuffle=True
    )
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    avg_loss = torch.tensor([]).to(device)
    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    train_test_dataset.train_dataset.labels = torch.nn.functional.one_hot(
        train_test_dataset.train_dataset.labels.to(torch.long), 2
    ).to(torch.float)

    # Freeze all layers for the plm but the last one
    if train_type == "direct_mlp":
        if hasattr(model, "plm") and hasattr(model.plm, "layers"):
            for i, layer in enumerate(model.plm.layers):
                for param in layer.parameters():
                    param.requires_grad = i == len(model.plm.layers) - 1

    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            if train_type == "direct_mlp":
                optimizer.zero_grad()
                a = model(x)
                total_loss = ce_loss_fn(a[2], y)
            elif train_type == "triplet":
                trips = torch.tensor(online_mine_triplets(y))

                if len(trips) <= 0:
                    continue

                hh = model(x)

                emb = torch.nn.functional.normalize(
                    hh[:, torch.tensor(pos_to_use), :], dim=1
                ).mean(dim=1)
                emb = torch.nn.functional.normalize(emb, dim=1)
                emb_trip = emb[trips]

                trip_loss = triplet_loss(
                    emb_trip[:, 0, :], emb_trip[:, 1, :], emb_trip[:, 2, :]
                )
                total_loss = trip_loss

            epoch_loss += total_loss.item()
            iter_20b_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            total_steps += 1
            running_batch_loss = torch.cat(
                [running_batch_loss, total_loss.detach().reshape(-1)]
            )

            if (step + 1) % 20 == 0:
                total_steps += 1
                iter_20b_loss = iter_20b_loss / 20
                running_20b_loss = torch.cat(
                    [running_20b_loss, iter_20b_loss.detach().reshape(-1)]
                )
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(
                    range(1, running_20b_loss.shape[0] + 1),
                    running_20b_loss.cpu().detach().numpy(),
                )
                plt.show()
            # print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, step, total_loss, trip_loss))
            # print(torch.unique(a[2].softmax(dim=1).argmax(dim=1), return_counts=True))
            print("[E%d I%d] %.3f " % (epoch, step, total_loss), flush=True)
            if total_steps % 1000 == 0:
                print("\t\tCheckpoint [%d]" % total_steps, flush=True)
                if save_models:
                    torch.save(
                        model.state_dict(),
                        checkpoints_dir + "/checkpoint_model_%d.pt" % total_steps,
                    )
                    torch.save(
                        running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt"
                    )
                    torch.save(
                        running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt"
                    )
                    torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")
        running_epoch_loss = torch.cat(
            [running_epoch_loss, epoch_loss.detach().reshape(-1)]
        )
    if save_models:
        torch.save(model.state_dict(), config["weights_path"])
        torch.save(running_batch_loss.cpu().detach(), f"{save_path}/batch_loss.pt")
        torch.save(running_epoch_loss.cpu().detach(), f"{save_path}/epoch_loss.pt")
        torch.save(running_20b_loss.cpu().detach(), f"{save_path}/20b_loss.pt")
        print(f"Model saved to {save_path}")
    return model


def train_epinnet(
    train_test_dataset,
    save_path: str,
    encodings=None,
    model=None,
    batch_size=32,
    iterations=20000,
    lr=1e-4,
    weight_decay=0.1,
    hidden_layers=[1024],
    activation="sigmoid",
    layer_norm=True,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
):
    torch.cuda.empty_cache()

    # Initialize model if not provided
    if model is None:
        if encodings is None:
            raise ValueError(
                "If model is not provided, encodings must be passed to initialize the model."
            )
        model = EpiNNet(
            encodings.shape[1],
            2,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device,
        ).to(device)

    # One-hot encode labels for training
    train_test_dataset.train_dataset.labels = torch.nn.functional.one_hot(
        train_test_dataset.train_dataset.labels.to(torch.long), 2
    ).to(torch.float)

    train_loader = torch.utils.data.DataLoader(
        train_test_dataset, batch_size=batch_size, shuffle=True
    )
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x)

            total_loss = ce_loss_fn(y_pred, y)
            epoch_loss += total_loss.item()
            iter_20b_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
            total_steps += 1

            running_batch_loss = torch.cat(
                [running_batch_loss, total_loss.detach().reshape(-1)]
            )

            if (step + 1) % 20 == 0:
                iter_20b_loss = iter_20b_loss / 20
                running_20b_loss = torch.cat(
                    [running_20b_loss, iter_20b_loss.detach().reshape(-1)]
                )
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(
                    range(1, running_20b_loss.shape[0] + 1),
                    running_20b_loss.cpu().detach().numpy(),
                )
                plt.draw()
                plt.pause(0.001)
                plt.close()

            print("[E%d I%d] %.3f" % (epoch, step, total_loss))
        running_epoch_loss = torch.cat(
            [running_epoch_loss, epoch_loss.detach().reshape(-1)]
        )

    # # Usage:
    # train_test_dataset.lazy_load_func()
    # train_test_dataset.test_dataset.labels = torch.nn.functional.one_hot(
    #     train_test_dataset.test_dataset.labels.to(torch.long), 2
    # ).to(torch.float)

    # eval_batch_size = 500
    # test_pred = torch.tensor([])
    # test_loader = torch.utils.data.DataLoader(
    #     train_test_dataset.test_dataset,
    #     batch_size=eval_batch_size,
    #     shuffle=False
    # )

    # test_score = torch.tensor([], dtype=torch.float)
    # with torch.no_grad():
    #     for i, batch in enumerate(test_loader):
    #         print("Evaluating test batch %d" % i)
    #         x = batch[0].to(device)
    #         y = batch[1].to(device)
    #         y_pred = model(x)
    #         test_pred = torch.cat([test_pred, y_pred.argmax(dim=1).cpu().detach()], dim=0)
    #         test_score = torch.cat([test_score, y_pred.softmax(dim=1)[:,0].cpu().detach()])

    # Save model

    # REVERT BACK LABELS
    train_test_dataset.train_dataset.labels = (
        train_test_dataset.train_dataset.labels.argmax(dim=1)
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), config["weights_path"])
    torch.save(running_batch_loss.cpu().detach(), f"{save_path}/batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), f"{save_path}/epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), f"{save_path}/20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model


def train_msa_backbone(
    plm_name: str,
    save_path: str,
    train_test_dataset,
    pos_to_use=None,
    batch_size=32,
    iterations=20000,
    margin=1.0,
    lr=4e-6,
    weight_decay=0.1,
    encoding_identifier=None,
    opmode="mean",
    hidden_layers=[1024],
    activation="sigmoid",
    layer_norm=False,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
    model=None  
):

    print("\n[DEBUG] Training parameters:")
    print(f"\tplm_name: {plm_name}")
    print(f"\tsave_path: {save_path}")
    print("\ttrain_dataset size: %d" % len(train_test_dataset.train_dataset))
    print(f"\tpos_to_use: {pos_to_use}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\titerations: {iterations}")
    print(f"\tmargin: {margin}")
    print(f"\tlr: {lr}")
    print(f"\tweight_decay: {weight_decay}")
    print(f"\tencoding_identifier: {encoding_identifier}")
    print(f"\topmode: {opmode}")
    print(f"\thidden_layers: {hidden_layers}")
    print(f"\tactivation: {activation}")
    
    print(f"\tlayer_norm: {layer_norm}")
    print(f"\tactivation_on_last_layer: {activation_on_last_layer}")
    print(f"\tdevice: {device}")
    print(f"\tmodel: {type(model)}\n")
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoints_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # In the future we might have other use cases,
    logits_only = True

    if model is None:
        model = plmTrunkModel(
            plm_name=plm_name,
            opmode=opmode,
            logits_only=logits_only,
            hidden_layers=hidden_layers,
            activation=activation,
            emb_only=False,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device
        ).to(device)
    else:
        print("Using provided model for training.")
        model = model.to(device)

    if train_test_dataset is None:
        raise ValueError("train_test_dataset must be provided as an argument.")

    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))
    print(f"Training for {n_epochs} epochs.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=CE_IGNORE_INDEX)

    model.train()
    avg_loss = torch.tensor([]).to(device)
    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    def noise_schedule(ranges, p=0.1):
        head_positions = []
        for r in ranges:
            r_list = list(r)
            n = len(r_list)
            if n == 0:
                continue
            probs = torch.tensor([1 - p, p])
            tosses = torch.multinomial(probs.repeat(n, 1), 1).squeeze(1)
            # Get positions where toss == 1 (heads)
            heads = [r_list[i] for i in range(n) if tosses[i].item() == 1]
            head_positions.extend(heads)
        return torch.tensor(head_positions, dtype=torch.long)

    ongoing_range_dict = {}

    def mine_ranges(ranges, indices):

        return_range_list = []
        for idx, global_idx in enumerate(indices):
            if global_idx in ongoing_range_dict:
                return_range_list.append(ongoing_range_dict[global_idx])
            else:
                rng = ranges[idx]
                rng = [x.split("_") for x in rng.split("x")]
                rng = [range(int(x[0]), int(x[1])) if len(x) == 2 else range(int(x[0]), int(x[0]) + 1) for x in rng]
                return_range_list.append(rng)
                ongoing_range_dict[global_idx] = rng

        return return_range_list


    mask_token = torch.tensor(model.tokenizer.encode("<mask>")).to(device)
        
    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)                  

            ranges = mine_ranges(train_test_dataset.train_dataset.sequence_dataframe["pad_regions"].iloc[y.cpu().numpy()].to_list(), y.cpu().numpy())
            
            # Apply random noise to the sequence
            mask_pos_matrix = torch.stack([torch.nn.functional.one_hot(noise_schedule(rng), x.shape[1]).sum(dim=0) for rng in ranges], dim=0)
            mask_pos_matrix = mask_pos_matrix.to(device)

            # Place masks instead of noise
            masked_sequence = (x - x * mask_pos_matrix) + mask_pos_matrix * mask_token

            logits = model(masked_sequence)

            # Get the ground truth labels on masked positiosn
            gt_labels = (((torch.ones(x.shape).to(device) - mask_pos_matrix) * CE_IGNORE_INDEX) + (x * mask_pos_matrix)).view(-1)
            total_loss = ce_loss_fn(logits.view(-1, len(model.vocab)), gt_labels.to(torch.long))

            epoch_loss += total_loss.item()
            iter_20b_loss += total_loss.item()

            total_loss.backward()        
            optimizer.step()

            total_steps += 1
            running_batch_loss = torch.cat([running_batch_loss, total_loss.detach().reshape(-1)])

            if (step + 1) % 20 == 0:
                total_steps += 1
                iter_20b_loss = iter_20b_loss /  20
                running_20b_loss = torch.cat([running_20b_loss, iter_20b_loss.detach().reshape(-1)])
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(range(1, running_20b_loss.shape[0] + 1), running_20b_loss.cpu().detach().numpy())
                plt.show()
            #print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, step, total_loss, trip_loss))
            # print(torch.unique(a[2].softmax(dim=1).argmax(dim=1), return_counts=True))
            print("[E%d I%d] %.3f " % (epoch, step, total_loss))
            if total_steps % 1000 == 0:
                print("\t\tCheckpoint [%d]" % total_steps)
                torch.save(model.state_dict(), checkpoints_dir + "/checkpoint_model_%d.pt" % total_steps)
                torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
                torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
                torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")

        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])

    torch.save(model.state_dict(), config["weights_path"])
    torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model


@torch.no_grad()
def epinnet_evaluate_function(
    model, data, aggregated_evaluated_data, device=torch.device("cpu")
):
    x = data[0].to(device)
    y = data[1].to(device)
    y = torch.nn.functional.one_hot(y.to(torch.long), 2).to(torch.float)
    y_pred = model(x)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    batch_loss = ce_loss_fn(y_pred, y)

        # Accumulate predictions, scores, true labels, and loss
    if "pred_label" not in aggregated_evaluated_data:
        aggregated_evaluated_data["pred_label"] = torch.tensor([], dtype=torch.long)
    if "active_prob" not in aggregated_evaluated_data:
        aggregated_evaluated_data["active_prob"] = torch.tensor([], dtype=torch.float)
    if "true_label" not in aggregated_evaluated_data:
        aggregated_evaluated_data["true_label"] = torch.tensor([], dtype=torch.long)
    if "loss" not in aggregated_evaluated_data:
        aggregated_evaluated_data["loss"] = torch.tensor([], dtype=torch.float)

    aggregated_evaluated_data["pred_label"] = torch.cat(
        [aggregated_evaluated_data["pred_label"], y_pred.argmax(dim=1).cpu().detach()],
        dim=0,
    )
    aggregated_evaluated_data["active_prob"] = torch.cat(
        [
            aggregated_evaluated_data["active_prob"],
            y_pred.softmax(dim=1)[:, 0].cpu().detach(),
        ]
    )
    aggregated_evaluated_data["true_label"] = torch.cat(
        [aggregated_evaluated_data["true_label"], y.argmax(dim=1).cpu().detach()]
    )
    aggregated_evaluated_data["loss"] = torch.cat(
        [aggregated_evaluated_data["loss"], batch_loss.detach().reshape(-1).cpu()]
    )

    # Flush the CUDA cache to free up unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return aggregated_evaluated_data


@torch.no_grad()
def epinnet_finalize_function(aggregated_evaluated_data, dataset):
    # Restore original finalize logic
    true_label = aggregated_evaluated_data["true_label"].cpu().detach()
    active_prob = aggregated_evaluated_data["active_prob"].cpu().detach()
    pred_label = aggregated_evaluated_data["pred_label"].cpu().detach()

    evaluated_df = pd.DataFrame(
        {
            "Score": active_prob,
            "GT": true_label,
            "Pred": pred_label,
            "Indices": dataset.indices,
        }
    )

    # top_K_df = pd.DataFrame(dict([("%d" % K, np.unique(true_label[np.argsort(-active_prob)[0:K]], return_counts=True)[1]) for K in [5,10,50,100,500,1000,5000]]))

    return {
        "evaluated_df": evaluated_df,
        "predicted_score": active_prob,
        "predicted_label": pred_label,
        "true_label": true_label,
        # "top_K_df": top_K_df,
    }


@torch.no_grad
def embeddings_evaluate_function(model, data, agg_dict, device=torch.device("cpu")):
    margin = 1
    pos_to_use = config["pos_to_use"]
    x = data[0].to(device)
    y = data[1].to(device)

    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)

    hh = model(x)

    emb = torch.nn.functional.normalize(hh[:, torch.tensor(pos_to_use), :], dim=1).mean(
        dim=1
    )
    emb = torch.nn.functional.normalize(emb, dim=1)

    if "trip_loss" not in agg_dict.keys():
        agg_dict["trip_loss"] = torch.tensor([], dtype=torch.float, device=device)

    if "embeddings" not in agg_dict.keys():
        agg_dict["embeddings"] = torch.tensor([], dtype=torch.float, device=device)

    if "ground_truth" not in agg_dict.keys():
        agg_dict["ground_truth"] = torch.tensor([], dtype=torch.float, device=device)

    trips = torch.tensor(online_mine_triplets(y))

    if len(trips) > 0:
        emb_trip = emb[trips]
        trip_loss = triplet_loss(
            emb_trip[:, 0, :], emb_trip[:, 1, :], emb_trip[:, 2, :]
        )
        agg_dict["trip_loss"] = torch.cat(
            [agg_dict["trip_loss"], trip_loss.detach().reshape(-1)], dim=0
        )

    agg_dict["embeddings"] = torch.cat([agg_dict["embeddings"], emb.detach()], dim=0)
    agg_dict["ground_truth"] = torch.cat(
        [agg_dict["ground_truth"], y.detach().reshape(-1)], dim=0
    )

    torch.cuda.empty_cache() 
    return agg_dict


@torch.no_grad()
def embeddings_finalize_function(agg_dict, dataset):
    embeddings = agg_dict["embeddings"].cpu().numpy()
    ground_truth = agg_dict["ground_truth"].cpu().numpy()

    # If embeddings is bigger than 15K, randomly sample 15K data points (by value, not by reference)
    max_points = 15000
    n_points = embeddings.shape[0]
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        embeddings = embeddings[idx].copy()
        ground_truth = ground_truth[idx].copy()

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        emb_3d[:, 0],
        emb_3d[:, 1],
        emb_3d[:, 2],
        c=ground_truth,
        cmap="viridis",
        alpha=0.7,
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_title("t-SNE of Embeddings (2D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.show()

    # Instead of showing the plot, return the figure object so it can be saved elsewhere
    plt.close(
        fig
    )  # Close the figure to prevent it from displaying in some environments
    agg_dict["tsne_visualization"] = fig

    # Compute pairwise cosine distance for all embeddings (no subsampling)
    all_embeddings = agg_dict["embeddings"].cpu()
    all_labels = agg_dict["ground_truth"].cpu().numpy()
    # Use the pairwise_cosine function defined at the top of the file
    distance_matrix = pairwise_cosine(all_embeddings).cpu().numpy()

    avg_dist_to_1 = []
    avg_dist_to_0 = []
    actual_label = []

    for i in range(distance_matrix.shape[0]):
        label_i = all_labels[i]
        # Indices for label 1 and label 0 (excluding self)
        idx_1 = np.where((all_labels == 1) & (np.arange(len(all_labels)) != i))[0]
        idx_0 = np.where((all_labels == 0) & (np.arange(len(all_labels)) != i))[0]
        # Average distances
        avg1 = np.mean(distance_matrix[i, idx_1]) if len(idx_1) > 0 else np.nan
        avg0 = np.mean(distance_matrix[i, idx_0]) if len(idx_0) > 0 else np.nan
        avg_dist_to_1.append(avg1)
        avg_dist_to_0.append(avg0)
        actual_label.append(label_i)

    df = pd.DataFrame(
        {
            "avg_dist_to_1": avg_dist_to_1,
            "avg_dist_to_0": avg_dist_to_0,
            "actual_label": actual_label,
            "indices": np.array(dataset.indices),
        }
    )

    # Overlay histograms of avg_dist_to_1 and avg_dist_to_0 for label 0 and label 1 separately

    # Subset for label 0
    df_0 = df[df["actual_label"] == 0]
    # Subset for label 1
    df_1 = df[df["actual_label"] == 1]

    # Plot for label 0: overlay avg_dist_to_1 and avg_dist_to_0
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    ax0.hist(
        df_0["avg_dist_to_1"].dropna(),
        bins=30,
        alpha=0.6,
        color="gray",
        label="Avg Dist to negatives",
    )
    ax0.hist(
        df_0["avg_dist_to_0"].dropna(),
        bins=30,
        alpha=0.6,
        color="green",
        label="Avg Dist to positives",
    )
    ax0.set_title("Positives: Avg Dist to negative and positives (Overlayed)")
    ax0.set_xlabel("Average Distance")
    ax0.set_ylabel("Frequency")
    ax0.legend()
    plt.tight_layout()

    agg_dict["hist_overlay_avg_dist_label0"] = fig0
    plt.close(fig0)

    # Plot for label 1: overlay avg_dist_to_1 and avg_dist_to_0
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.hist(
        df_1["avg_dist_to_1"].dropna(),
        bins=30,
        alpha=0.6,
        color="gray",
        label="Avg Dist to negatives",
    )
    ax1.hist(
        df_1["avg_dist_to_0"].dropna(),
        bins=30,
        alpha=0.6,
        color="green",
        label="Avg Dist to positives",
    )
    ax1.set_title("Negatives: Avg Dist to negatives and positives (Overlayed)")
    ax1.set_xlabel("Average Distance")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    plt.tight_layout()
    agg_dict["hist_overlay_avg_dist_label1"] = fig1
    plt.close(fig1)
    agg_dict["pairwise_distance_df"] = df

    return agg_dict


def train_evaluate_plms():
    ref_seq = config["ref_seq"]

    # Check if "plm_name" exists in config, else use default "esm2_t12_35M_UR50D"
    if "plm_name" in config and config["plm_name"]:
        plm_name = config["plm_name"]
    else:
        print(
            "Warning: 'plm_name' not found in config. Using default 'esm2_t12_35M_UR50D'."
        )
        plm_name = "esm2_t12_35M_UR50D"

    train_indices_func = (
        lambda sdf: list(sdf.index)
        if config["train_indices"] == "all"
        else get_indices(
            sdf,
            config["train_indices"],
            nmuts_column=config["nmuts_column"],
            rev=config["train_indices_rev"],
        )
    )

    test_indices_func = lambda sdf: get_indices(
        sdf,
        config["test_indices"],
        nmuts_column=config["nmuts_column"],
        rev=config["test_indices_rev"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ֿ\t\t[INFO] Using device: {device}")

    emb_only = config["train_type"] == "triplet"

    model = plmTrunkModel(
        plm_name=plm_name,
        opmode="mean",
        emb_only=emb_only,
        hidden_layers=[256, 128],
        activation="relu",
        layer_norm=False,
        activation_on_last_layer=False,
        specific_pos=config["pos_to_use"],
        device=device,
    ).to(device)

    # Load weights for plm backbone only if config has load_weights set to True
    if "load_weights" in config and config["load_weights"]:
        backbone_weights = torch.load(config["weights_path"], map_location=device)
        backbone_weights = {
            k.replace("plm.", "", 1): v
            for k, v in backbone_weights.items()
            if "plm." in k
        }
        model.plm.load_state_dict(backbone_weights, strict=True)

    train_test_dataset = EpiNNetActivityTrainTest(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=train_indices_func,
        test_indices=test_indices_func,
        encoding_function=model.encode,
        encoding_identifier=plm_name,
        cache=True,
        lazy_load=True,
        sequence_column_name=config["sequence_column_name"],
        activity_column_name=config["activity_column_name"],
        ref_seq=ref_seq,
        labels_dtype=torch.float32,
        device=device,
    )

    if config["train"]:
        model.plm.token_dropout = config["train_drop_tokens"]

        model = train_plm_triplet_model(
            plm_name=plm_name,
            train_type=config["train_type"],
            save_path=config["save_path"],
            train_test_dataset=train_test_dataset,
            pos_to_use=config["pos_to_use"],
            batch_size=config["batch_size"],
            iterations=config["iterations"],
            lr=float(config["lr"]),
            device=device,
            model=model,
        )

    model.plm.token_dropout = config["inference_drop_tokens"]

    if config["evaluate_train"] or config["evaluate_test"]:
        train_test_dataset.evaluate(model,
                                    embeddings_evaluate_function, 
                                    embeddings_finalize_function,
                                    eval_train=config["evaluate_train"],
                                    eval_test=config["evaluate_test"],
                                    internal_batch_size=config["batch_size"])

    if config["evaluate_train"] and config["evaluate_test"]:
        # Train and evaluate simple trunk
        from simple_mlp_fit import train_trunk_mlp

        train_trunk_mlp(config["save_path"])


def bootstrap_train_plms():
    print(f"Bootstrapping training with config: {config}")
    ref_seq = config["ref_seq"]

    # Check if "plm_name" exists in config, else use default "esm2_t12_35M_UR50D"
    if "plm_name" in config and config["plm_name"]:
        plm_name = config["plm_name"]
    else:
        print(
            "Warning: 'plm_name' not found in config. Using default 'esm2_t12_35M_UR50D'."
        )
        plm_name = "esm2_t12_35M_UR50D"

    if config["bootstrap_index"] is None:
        print("No bootstrap index give, going through all")
        bootstrap_indices = range(config["n_bootstrap"])
    else:
        print(f"Training bootstrap index {config["bootstrap_index"]}")
        bootstrap_indices = [config["bootstrap_index"]]

    for i in bootstrap_indices:
        with open(f"{config["bootstrap_path"]}/{config["bootstrap_indices_prefix"]}_{i}.json", "r") as f:
            indices_dict = json.load(f)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ֿ\t\t[INFO] Using device: {device}")

        emb_only = config["train_type"] == "triplet"

        model = plmTrunkModel(
            plm_name=plm_name,
            opmode="mean",
            emb_only=emb_only,
            hidden_layers=[256, 128],
            activation="relu",
            layer_norm=False,
            activation_on_last_layer=False,
            specific_pos=config["pos_to_use"],
            device=device,
        ).to(device)

        # Load weights for plm backbone only if config has load_weights set to True
        if "load_weights" in config and config["load_weights"]:
            backbone_weights = torch.load(config["weights_path"], map_location=device)
            backbone_weights = {
                k.replace("plm.", "", 1): v
                for k, v in backbone_weights.items()
                if "plm." in k
            }
            model.plm.load_state_dict(backbone_weights, strict=True)

        train_test_dataset = EpiNNetActivityTrainTest(
            train_project_name="triplet_training",
            evaluation_path=config["save_path"],
            dataset_path=config["dataset_path"],
            train_indices=indices_dict["train"],
            test_indices=indices_dict["test"],
            encoding_function=model.encode,
            encoding_identifier=plm_name,
            cache=True,
            lazy_load=True,
            sequence_column_name=config["sequence_column_name"],
            activity_column_name=config["activity_column_name"],
            ref_seq=ref_seq,
            labels_dtype=torch.float32,
            device=device,
        )

        model.plm.token_dropout = config["train_drop_tokens"]

        model = train_plm_triplet_model(
            plm_name=plm_name,
            train_type=config["train_type"],
            save_path=config["save_path"],
            train_test_dataset=train_test_dataset,
            pos_to_use=config["pos_to_use"],
            batch_size=config["batch_size"],
            iterations=config["iterations"],
            lr=float(config["lr"]),
            device=device,
            model=model,
            save_models=False
        )

        # save embeddings of dataset
        dataset = EpiNNetDataset(
            config["dataset_path"],
            indices=None,
            encoding_function=model.encode,
            encoding_identifier=plm_name,
        )
        x, y = dataset[:]
        x = x.to(device)
        y = y.to(device)
        hh = model(x)
        emb = torch.nn.functional.normalize(hh[:, torch.tensor(config["pos_to_use"]), :], dim=1)
        torch.save(emb.cpu().detach(), f"{config["bootstrap_path"]}embeddings_{i}.pt")


def train_evaluate_epinnet():
    # Define train/test split functions
    train_indices_func = lambda sdf: get_indices(
        sdf,
        config["train_indices"],
        nmuts_column=config["nmuts_column"],
        rev=config["train_indices_rev"],
    )

    test_indices_func = lambda sdf: get_indices(
        sdf,
        config["test_indices"],
        nmuts_column=config["nmuts_column"],
        rev=config["test_indices_rev"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the sequence dataframe and create encodings
    sequence_df = pd.read_csv(config["dataset_path"])
    encodings = torch.tensor(
        get_one_hot_encoding(
            sequence_df, config["first_column_name"], config["last_column_name"]
        ),
        dtype=torch.float,
    )

    # Initialize EpiNNet model
    epinnet_model = EpiNNet(
        d_in=encodings.shape[1],  # Example input dimension, adjust as needed
        hidden_layers=[512, 256],  # Example hidden layers, adjust as needed
        d_out=2,  # Example output dimension for binary classification
        activation="relu",
        device=device,
    ).to(device)

    if "load_weights" in config and config["load_weights"]:
        backbone_weights = torch.load(config["weights_path"], map_location=device)
        epinnet_model.load_state_dict(backbone_weights)

    train_test_dataset = EpiNNetActivityTrainTest(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=train_indices_func,
        test_indices=test_indices_func,
        encoding_function=None,
        encoding_identifier=None,
        external_encoding=encodings,
        cache=True,
        lazy_load=True,
        sequence_column_name="full_seq",
        activity_column_name="inactive",
        ref_seq=config["ref_seq"],
        labels_dtype=torch.float32,
        device=device,
        hidden_layers=config["hidden_layers"],
    )

    if config["train"]:
        epinnet_model = train_epinnet(
            train_test_dataset=train_test_dataset,
            save_path=config["save_path"],
            encodings=encodings,
            model=epinnet_model,
            device=device,
            lr=float(config["lr"]),
            iterations=int(config["iterations"]),
            batch_size=int(config["batch_size"]),
        )

    if config["evaluate_train"] or config["evaluate_test"]:
        train_test_dataset.evaluate(
            epinnet_model,
            epinnet_evaluate_function,
            epinnet_finalize_function,
            eval_train=config["evaluate_train"],
            eval_test=config["evaluate_test"],
        )


def train_evaluate_msa_backbone():
    # df = pd.read_csv(config["dataset_path"])
    # df["pad_regions"] = df["full_seq"].apply(get_non_alignment_regions)
    # df.to_csv(config["dataset_path"], index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if "plm_name" exists in config, else use default "esm2_t12_35M_UR50D"
    if "plm_name" in config and config["plm_name"]:
        plm_name = config["plm_name"]
    else:
        print("Warning: 'plm_name' not found in config. Using default 'esm2_t12_35M_UR50D'.")
        plm_name = "esm2_t12_35M_UR50D"

    msa_backbone_model =\
     plmTrunkModel(
        plm_name=plm_name,
        opmode="mean",
        emb_only = False,
        logits_only = True,
        hidden_layers=config["hidden_layers"],
        activation="relu",
        layer_norm=False,
        activation_on_last_layer=False,
        device=device
    ).to(device)

    print(f"ֿ\t\t[INFO] Using device: {device}")
    print(f"ֿ\t\t[INFO] Using hidden layers: {config['hidden_layers']}")
    print(f"ֿ\t\t[INFO] Using plm: {plm_name}")
    print(f"ֿ\t\t[INFO] Using dataset: {config['dataset_path']}")
    print(f"ֿ\t\t[INFO] Using save path: {config['save_path']}")
    print(f"ֿ\t\t[INFO] Using weights path: {config['weights_path']}")
    print(f"ֿ\t\t[INFO] Using batch size: {config['batch_size']}")
    print(f"ֿ\t\t[INFO] Using iterations: {config['iterations']}")
    print(f"ֿ\t\t[INFO] Using learning rate: {config['lr']}")
    print(f"ֿ\t\t[INFO] Using train type: {config['train_type']}")

    def encode_aligned_seq(seq):
        seq = seq.replace("-", "<pad>")
        return msa_backbone_model.tokenizer.encode(seq)

    # Prepare dataset (assuming similar interface as EpiNNetActivityTrainTest)
    train_test_dataset = EpiNNetActivityTrainTest(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=None,
        test_indices=None,
        encoding_function=encode_aligned_seq,
        encoding_identifier=plm_name,
        cache=True,
        lazy_load=True,
        sequence_column_name=config["sequence_column_name"],
        activity_column_name=config["activity_column_name"],
        ref_seq=config["ref_seq"],
        labels_dtype=torch.float32,
        device=device
    )

    if config["train"]:
        msa_backbone_model = train_msa_backbone(
                plm_name=plm_name,
                train_test_dataset=train_test_dataset,
                save_path=config["save_path"],                
                batch_size=config["batch_size"],
                iterations=config["iterations"],
                lr=float(config["lr"]),
                device=device,
                model=msa_backbone_model,
                hidden_layers=config["hidden_layers"] 
            )


       
    if config["evaluate_train"] or config["evaluate_test"]:
        train_test_dataset.evaluate(
            msa_backbone_model,
            #msa_backbone_evaluate_function,
            #msa_backbone_finalize_function,
            eval_train=config.get("evaluate_train", False),
            eval_test=config.get("evaluate_test", False)
        )


if __name__ == "__main__":
    if config["model_type"] == "plm":
        if config["train_type"] == "msa_backbone":
            train_evaluate_msa_backbone()
        elif config["train_type"] == "bootstrap":
            bootstrap_train_plms()
        else:
            train_evaluate_plms()
    elif config["model_type"] == "epinnet":
        train_evaluate_epinnet()
