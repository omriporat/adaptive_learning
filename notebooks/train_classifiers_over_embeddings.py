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
from sklearn.ensemble import GradientBoostingClassifier

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

embedding_paths = ["%s/data/gfp/embeddings/esm_650m" % base_path,
                    "%s/data/gfp/embeddings/esm_35m" % base_path,
                    "%s/data/gfp/embeddings/esm_8m" % base_path]


classifier_embeddings_path = embedding_paths[1]
train_data = {}


for n_train in range(1, 6):
    n_train_data = n_train
    n_test_data = list(range(n_train + 1, 11))

    for i in range(1, n_train_data + 1):
        labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % i))
        indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % i))
        embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % i))

        train_data["nmuts_%d" % i] = {
            "labels": labels,
            "indices": indices,
            "embeddings": embeddings
        }
        
        
    indices_all = []
    labels_all = []
    embeddings_all = []

    for k, v in train_data.items():
        indices_all.append(v["indices"])
        labels_all.append(v["labels"])
        embeddings_all.append(v["embeddings"])

    indices_all = torch.cat(indices_all)
    embeddings_all = torch.cat(embeddings_all)
    labels_all = torch.cat(labels_all)


    mlp_base_parameters = {
        "activation" : 'relu',           
        "solver" : 'lbfgs', 
        "batch_size": 128,   
        "alpha" : 1,                
        "learning_rate_init" : 1e-3,    
        "max_iter" :200,
        "random_state" : 4321,                
        "early_stopping" : True,         
        "n_iter_no_change" : 10,         
        #"random_state" : 42,
        "verbose": True
    }

    flat_embeddings = embeddings_all.mean(axis=1)
    normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
    normalized_embeddings = normalized_embeddings / flat_embeddings.std(dim=0, keepdim=True)

    print("Fitting MLP over %dx%d embeddings to %d labels" % (normalized_embeddings.shape[0], normalized_embeddings.shape[1], len((labels_all))))

    mlp_llm = MLPClassifier(hidden_layer_sizes=(64,), **mlp_base_parameters)
    mlp_llm.fit(normalized_embeddings.numpy(), labels_all)

    

    # print("Fitting GradientBoostingClassifier over %dx%d embeddings to %d labels" % (normalized_embeddings.shape[0], normalized_embeddings.shape[1], len((labels_all))))

    # gbc = GradientBoostingClassifier(
    #     n_estimators=64,
    #     learning_rate=0.1,
    #     max_depth=3,
    #     random_state=4321,
    #     verbose=1
    # )


    # gbc.fit(normalized_embeddings.numpy(), labels_all)
    # mlp_llm = gbc  # For downstream code to use the same variable name



    test_data = []

    for i in n_test_data:
        labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % i))
        indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % i))
        embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % i))

        # test_data["nmuts_%d" % i] = {
        #     "labels": labels,
        #     "indices": indices,
        #     "embeddings": embeddings
        # }

        flat_embeddings = embeddings.mean(axis=1)
        normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
        normalized_embeddings = normalized_embeddings / flat_embeddings.std(dim=0, keepdim=True)

        predictions_proba = mlp_llm.predict_proba(normalized_embeddings.numpy())
        predictions = (predictions_proba[:, 1] > 0.5).astype(int)
        #predictions = mlp_llm.predict(normalized_embeddings.numpy())
        print(predictions_proba)
        print(predictions)
        #print((predictions_proba[:, 1] > 0.5).astype(int))
        

        evaluation = evaluate_classifier(predictions_proba[:, 1], predictions, labels.numpy())

        print("Evaluation for %d mutations: %s" % (i, evaluation))

        test_data.append(evaluation)

        evaluation["test_mutations"] = i
        evaluation["train_mutations"] = n_train_data
        evaluation["classifier"] = classifier_embeddings_path


    os.makedirs("results/classification_results/embeddings/gradient_boosting", exist_ok=True)
    test_data = pd.DataFrame(test_data)
    print(test_data)
    test_data.to_csv("results/classification_results/embeddings/gradient_boosting/35m_evaluation_train_on_%d.csv" % n_train_data, index=False)
