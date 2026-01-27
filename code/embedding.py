from typing import List

import numpy as np
import torch

from dataset import PREDataset
from plm_base import plm_init
from models import plmEmbeddingModel
from config import load_config


def load_model(
    plm_name: str,
    state_dict_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = plmEmbeddingModel(
        plm_name=plm_name,
        emb_only=True,
        logits_only=False,
        device=device
    ).to(device)

    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path))

    return model


def load_dataset(dataset_path: str, model: torch.nn.Module, cache: bool = True):
    dataset = PREDataset(
        dataset_path=dataset_path,
        indices=None,
        cache=cache,
        encoding_function=model.encode,
        encoding_identifier="plm_embedding",
    )
    return dataset


def get_embeddings(dataset: PREDataset,model: torch.nn.Module, positions: List[int], opmode: str = "mean"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    emb = []
    ys = []

    for step, batch in enumerate(loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        hh = model(x)

        if opmode == "mean":
            hh = torch.nn.functional.normalize(
                hh[:, torch.tensor(positions), :], dim=1
            ).mean(dim=1)
        elif opmode == "flat":
            # select specific positions and flatten
            hh = hh[:, torch.tensor(positions), :].reshape(hh.size(0), -1)
        else:
            raise ValueError("Unsupported opmode %s" % opmode)
        
        emb.append(hh.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        print(f"Processed batch {step + 1}/{len(loader)}")

    emb = np.concatenate(emb, axis=0)
    ys = np.concatenate(ys, axis=0)

    return emb, ys


def main():
    # Load configuration from YAML file
    config = load_config("configs/config_original_paper.yaml")
    plm_init(config["root_path"])
    enzyme = config["enzyme"]
    finetune = config.get("finetune", False)
    opmode = config.get("opmode", "mean")
    if finetune:
        model = load_model(
            plm_name=config["plm_name"],
            pos_to_use=config["pos_to_use"],
            state_dict_path=config["weights_path"],
        )
    else:
        model = load_model(plm_name=config["plm_name"])
    dataset = load_dataset(dataset_path=config["dataset_path"], model=model)
    embeddings, labels = get_embeddings(dataset=dataset,model=model, positions=config["pos_to_use"], opmode=opmode)
    # save embeddings
    np.save(config["embeddings_path"], embeddings)


if __name__ == "__main__":
    main()
