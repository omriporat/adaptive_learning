
import sys
import os
sys.path.append(os.path.join(os.path.abspath(''), 'code'))

import pandas as pd
from config import load_config

from Bio import SeqIO

DPTE2_FILE = "dPTE2.txt"
PREFIX = "ITNSG"

def convert_sequence_to_dpte2(df: pd.DataFrame, dpte2_seq: str, config: dict) -> pd.DataFrame:
    dpte2_values = []
    for _, row in df.iterrows():
        mutated_seq = list(dpte2_seq)
        designed_positions = row[config["first_column_name"]:config["last_column_name"]].values
        for pos, mut in zip(config["pos_to_use"], designed_positions):
            mutated_seq[pos - 1] = mut
        dpte2_values.append("".join(mutated_seq))

    df["dPTE2_sequence"] = dpte2_values

    # verify with regular sequence
    for i, row in df.iterrows():
        dpte2_seq = row["dPTE2_sequence"]
        regular_seq = row["full_seq"]
        for pos in config["pos_to_use"]:
            if dpte2_seq[pos - 1] != regular_seq[pos - 1]:
                raise ValueError(f"Mismatch at position {pos} for sequence {i}")

    return df


def export_to_fasta(df: pd.DataFrame, output_path: str):
    with open(output_path, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{i} - {row['serial_number']}\n")
            seq = f"{PREFIX}{row['dPTE2_sequence']}\n"
            # break the sequence into lines of 80 characters            
            for j in range(0, len(seq), 80):
                f.write(seq[j:j+80])
                f.write("\n")


def verify_fasta(output_path: str, df: pd.DataFrame, config: dict):
    for record in SeqIO.parse(output_path, "fasta"):
        print(f"Verifying record {record.id} with description {record.description}")
        seq_id = record.id
        serial_number = record.description.split(" - ")[1]
        row = df[df["serial_number"] == serial_number]
        if row.empty:
            raise ValueError(f"Serial number {serial_number} not found in DataFrame")
        if row.index[0] != int(seq_id):
            raise ValueError(f"Sequence ID {seq_id} does not match DataFrame index {row.index[0]}")
        original_seq = row["dPTE2_sequence"].values[0]
        if str(record.seq) != f"{PREFIX}{original_seq}":
            raise ValueError(f"Mismatch for serial number {serial_number}")
        
        designed_pos_aa = [record.seq[pos - 1 + len(PREFIX)] for pos in config["pos_to_use"]]
        print(f"Designed positions: designed pos sequence: {designed_pos_aa}\n")
            

def main():
    config = load_config("configs/config_PTE.yaml")
    dpte2_path = f"{config['data_dir_path']}/{DPTE2_FILE}"
    with open(dpte2_path, "r") as f:
        dpte2_seq = f.read().strip()

    if config.get("delta_embeddings", False):
        delta_embedding_tag = "delta_embeddings"
    else:
        delta_embedding_tag = "regular_embeddings"
    
    representatives_df_path = f"{config['results_path']}/cluster_representative_sequences_{config['opmode']}_{config['finetune_tag']}_{config['cluster_method']}_{delta_embedding_tag}.csv"
    representatives_df = pd.read_csv(representatives_df_path, dtype={"serial_number": str})
    dpte2_df = convert_sequence_to_dpte2(representatives_df, dpte2_seq, config)

    output_path = f"{config['results_path']}/cluster_representative_sequences_{config['opmode']}_{config['finetune_tag']}_{config['cluster_method']}_{delta_embedding_tag}_dpte2.csv"
    dpte2_df.to_csv(output_path, index=False)

    fasta_output_path = f"{config['results_path']}/cluster_representative_sequences_{config['opmode']}_{config['finetune_tag']}_{config['cluster_method']}_{delta_embedding_tag}_dpte2.fasta"
    export_to_fasta(dpte2_df, fasta_output_path)
    verify_fasta(fasta_output_path, dpte2_df, config)


if __name__ == "__main__":
    main()
