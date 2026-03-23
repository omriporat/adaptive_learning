
import os

CONFIG_FILE = "configs/config_original_paper.yaml"
CMD = "bsub -q short -oo ./job_logs/bootstrap_{substrate}_{opmode}_{finetune_tag}_{train_fraction}_{log_tag}_{delta_embedding_tag}.log -R \"rusage[mem=2GB]\" \"source .venv/bin/activate; python3 code/bootstrap.py {config_file} --substrate {substrate} --opmode {opmode} --{finetune_tag} --train_fraction {train_fraction} --n_bootstrap 1000 --{log_tag} --{delta_embedding_tag}\""
SUBSTRATES = [
    "naphthyl_acetate",
    "p-nitrophenyl_acetate",
    "p-nitrophenyl_octanoate",
    "malathion",
    "paraoxon",
    "nonanoic_lactone",]
OPMODES = ["mean"]
TRAIN_FRACTIONS = [0.6, 0.7, 0.8]
FINETUNE_TAGS = ["no-finetune"]
LOG_TAGS = ["no-log_for_bootstrap", "log_for_bootstrap"]
DELTA_EMBEDDING_TAGS = ["no-delta_embeddings"]


def main():
    for substrate in SUBSTRATES:
        for opmode in OPMODES:
            for finetune_tag in FINETUNE_TAGS:
                for train_fraction in TRAIN_FRACTIONS:
                    for log_tag in LOG_TAGS:
                        for delta_embedding_tag in DELTA_EMBEDDING_TAGS:
                            cmd = CMD.format(
                                config_file=CONFIG_FILE,
                                substrate=substrate,
                                opmode=opmode,
                                finetune_tag=finetune_tag,
                                train_fraction=train_fraction,
                                log_tag=log_tag,
                                delta_embedding_tag=delta_embedding_tag,
                        )
                            os.system(cmd)


if __name__ == "__main__":
    main()