import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
# aggregate all the data

base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)

    return(one_hot_encoding)
    
df = pd.read_csv("data/nmt/nmt_full_seq.csv")



base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

embedding_paths = ["%s/data/nmt/embeddings/esm_35m" % base_path]


classifier_embeddings_path = embedding_paths[0]

si = np.where(df.columns == "1")[0][0]
ei = np.where(df.columns == "272")[0][0]+1


positions_with_mutations =  np.array([len(pd.unique(df.iloc[:,i])) > 1 for i in range(si,ei)])


new_df_columns = zip(df.columns[si:ei][positions_with_mutations],
                            df.iloc[0,si:ei][positions_with_mutations])


new_df_columns =["%s%s" % (b,a) for a,b in new_df_columns]     


new_df =\
    pd.concat([df["name"],
            df["seq"],
            df["activity"], 
            df["num_muts"],
            df["p1"],
            df["p2"],
            df.iloc[:,si:ei].iloc[:,positions_with_mutations]
            ], axis=1)


new_df.columns = ['name', 'seq', 'activity', 'num_muts', 'p1', 'p2'] + new_df_columns
print(new_df)

one_hot = get_one_hot_encoding(new_df, "Y20", "F253")
si = np.where(new_df.columns == "Y20")[0][0]
ei = np.where(new_df.columns == "F253")[0][0]
assert one_hot.shape[1] == sum([len(pd.unique(new_df[C])) for C in new_df.columns[si:(ei+1)]])

labels = torch.load(os.path.join(classifier_embeddings_path, "y_values.pt"))
indices = torch.load(os.path.join(classifier_embeddings_path, "indices.pt"))
embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings.pt"))

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
    "verbose": False
}

flat_embeddings = embeddings.mean(axis=1)
normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
normalized_embeddings = normalized_embeddings / flat_embeddings.std(dim=0, keepdim=True)

print("Fitting MLP over %dx%d embeddings to %d labels" % (normalized_embeddings.shape[0], normalized_embeddings.shape[1], len((labels))))
print("Fitting MLP over %dx%d one-hot encodings to %d labels" % (one_hot.shape[0], one_hot.shape[1], len((labels))))

original_labels = labels.clone()

all_results = []
for frac in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    for iter in range(10):
        # INSERT_YOUR_CODE
        labels = original_labels[torch.randperm(original_labels.size(0))]

        N_samples = int(frac * one_hot.shape[0])
        train_indices = np.random.choice(one_hot.shape[0], N_samples, replace=False)
        test_indices = np.setdiff1d(np.arange(one_hot.shape[0]), train_indices)

        #assert(sum(new_df["activity"].to_numpy() == labels.numpy()) == one_hot.shape[0])

        mlp_llm = MLPRegressor(hidden_layer_sizes=(64,), **mlp_base_parameters)
        mlp_llm.fit(normalized_embeddings.numpy()[train_indices], labels.numpy()[train_indices])

        mlp_ohe = MLPRegressor(hidden_layer_sizes=(64,), **mlp_base_parameters)
        mlp_ohe.fit(one_hot.numpy()[train_indices], labels.numpy()[train_indices])

        cor_llm = spearmanr(mlp_llm.predict(normalized_embeddings.numpy()[test_indices]), labels.numpy()[test_indices])
        cor_ohe = spearmanr(mlp_ohe.predict(one_hot.numpy()[test_indices]), labels.numpy()[test_indices])

        result_dict = {"frac": frac, "iter": iter, "cor_llm": cor_llm.correlation, "cor_ohe": cor_ohe.correlation}
        print(result_dict)
        all_results.append(result_dict)

        result_df = pd.DataFrame(all_results)
        print(result_df)

        result_df.to_csv("data/nmt/second_shuffle_across_fraction_evaluations.csv", index=False)

# plt.scatter(mlp_llm.predict(normalized_embeddings.numpy()[test_indices]), labels.numpy()[test_indices])


# plt.title("MLP over ESM: %f (Spearman rho)" % cor.correlation)
# plt.xlabel("Predicted activity")
# plt.ylabel("True activity")


# plt.scatter(mlp_ohe.predict(one_hot.numpy()[test_indices]), labels.numpy()[test_indices])
# plt.title("MLP over OHE: %f (Spearman rho)" % cor.correlation)
# plt.xlabel("Predicted activity")
# plt.ylabel("True activity")