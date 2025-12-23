import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt

def evaluate_classifier(score, 
                        predicted_label, 
                        gt_label,
                        label_true=0,
                        label_false=1):

    tp = sum((predicted_label == label_true) & (gt_label == label_true))
    tn = sum((predicted_label == label_false) & (gt_label == label_false))
    fp = sum((predicted_label == label_true) & (gt_label == label_false))
    fn = sum((predicted_label == label_false) & (gt_label == label_true))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(predicted_label == gt_label) / len(predicted_label)
    roc = roc_auc_score(gt_label, score)

    ordered_scores = np.argsort(score)[0:100]
    top_100_pct = sum(gt_label[ordered_scores] == label_true) / 100

    evaluation = {
        "tp" : tp,
        "tn" : tn,
        "fp" : fp,
        "fn" : fn,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "accuracy" : accuracy,
        "roc" : roc,
        "top_100_pct": top_100_pct
    }
    #roc = sklearn.
    return evaluation
    


# aggregate all the data

base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)

    return(one_hot_encoding)
    
df = pd.read_csv("data/gfp_dataset_10mut.csv")
one_hot = get_one_hot_encoding(df, "L42", "V224")
si = np.where(df.columns == "L42")[0][0]
ei = np.where(df.columns == "V224")[0][0]
assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in df.columns[si:(ei+1)]])



for n_train in range(1, 6):

    n_train_data = n_train
    n_test_data = list(range(n_train + 1, 11))

    train_indices = (df["num_muts"] <= n_train) & (df["num_muts"] > 0)


    mlp_base_parameters = {
        "activation" : 'relu',           
        "solver" : 'lbfgs', 
        "batch_size": 128,   
        "alpha" : 1,                
        "learning_rate_init" : 1e-3,    
        "max_iter" :500,
        "random_state" : 4321,                
        "early_stopping" : True,         
        "n_iter_no_change" : 200,         
        #"random_state" : 42,
        "verbose": True
    }

    train_ohe = one_hot[train_indices]
    train_labels = df.inactive.astype(int)[train_indices]


    print("Fitting MLP over %dx%d embeddings to %d labels" % (train_ohe.shape[0], train_ohe.shape[1], len((train_labels))))

    mlp_ohe = MLPClassifier(hidden_layer_sizes=(200,), **mlp_base_parameters)
    mlp_ohe.fit(train_ohe.numpy(), train_labels.to_numpy())



    test_data = []

    for i in n_test_data:
        test_indices = df["num_muts"] == i
        
        test_ohe = one_hot[test_indices]
        test_labels = df.inactive.astype(int)[test_indices]

        predictions_proba = mlp_ohe.predict_proba(test_ohe.numpy())
        predictions = (predictions_proba[:, 1] > 0.5).astype(int)

        evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, test_labels.to_numpy())
        print("Evaluation for %d mutations: %s" % (i, evaluation))

        test_data.append(evaluation)

        evaluation["test_mutations"] = i
        evaluation["train_mutations"] = n_train_data
        evaluation["classifier"] = "ohe"


    test_data = pd.DataFrame(test_data)
    test_data.to_csv("./results/classification_results/ohe/200/ohe_evaluation_train_on_%d.csv" % n_train_data, index=False)
    print(test_data)