import pandas as pd
import numpy as np
import os
import torch

df = pd.read_csv("data/gfp_dataset_10mut.csv")

base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"
intermediate_result_paths = [#"%s/results/gfp_embeddings/esm_650m/evaluations" % base_path,
                            "%s/results/gfp_embeddings/esm_35m/evaluations" % base_path,
                            "%s/results/gfp_embeddings/esm_8m/evaluations" % base_path]

final_save_path = [#"%s/data/gfp/embeddings/esm_650m" % base_path,
                    "%s/data/gfp/embeddings/esm_35m" % base_path,
                    "%s/data/gfp/embeddings/esm_8m" % base_path]

hidden_dim_size = [#1280, 
480, 320]

df = pd.read_csv("data/gfp_dataset_10mut.csv")

for intermediate_result_path, final_save_path, hs in zip(intermediate_result_paths, final_save_path, hidden_dim_size):

    print("Processing %s and saving to %s" % (intermediate_result_path, final_save_path))

    subfolders = [f.name for f in os.scandir(intermediate_result_path) if f.is_dir()]

    embedding_all = torch.zeros([df.shape[0], 22, hs], dtype=torch.float)
    label_all = torch.zeros([df.shape[0]], dtype=torch.float)
    indices_all = torch.zeros([df.shape[0]], dtype=torch.int64)

    for i, subfolder in enumerate(subfolders):

        print("\tLoading %s [%d/%d]" % (subfolder, i, len(subfolders)))
        embeddings = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "embeddings.pt"))
        embeddings = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "embeddings.pt"))
        labels = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "y_value.pt"))
        indices = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "indices.pt"))
        
        indices_all[indices] = indices
        label_all[indices] = labels.to(torch.float)
        embedding_all[indices] = embeddings

    for i in range(1, 11):
        slice_indices = np.where(df["num_muts"] == i)[0]
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i), i, 10))
        torch.save(embedding_all[slice_indices], os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i), i, 10))
        torch.save(label_all[slice_indices], os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i), i, 10))
        torch.save(indices_all[slice_indices], os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i))
    
    del embedding_all, label_all, indices_all
