
import os

CONFIG_FILE = "configs/config_original_paper.yaml"
CMD = "bsub -q short -oo ./job_logs/bootstrap_{substrate}_{opmode}_{finetune_tag}_{train_fraction}_{log_tag}.log -R \"rusage[mem=4GB]\" \"source .venv/bin/activate; python3 code/bootstrap.py {config_file} --substrate {substrate} --opmode {opmode} --{finetune_tag} --train_fraction {train_fraction} --n_bootstrap 1000 --{log_tag}\""
SUBSTRATES = [
    "naphthyl_acetate",
    "p-nitrophenyl_acetate",
    "p-nitrophenyl_octanoate",
    "malathion",
    "paraoxon",
    "nonanoic_lactone",]
OPMODES = ["mean"]
TRAIN_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
FINETUNE_TAGS = ["no-finetune"]
LOG_TAGS = ["no-log_for_bootstrap", "log_for_bootstrap"]


def main():
    for substrate in SUBSTRATES:
        for opmode in OPMODES:
            for finetune_tag in FINETUNE_TAGS:
                for train_fraction in TRAIN_FRACTIONS:
                    for log_tag in LOG_TAGS:
                        cmd = CMD.format(
                            config_file=CONFIG_FILE,
                            substrate=substrate,
                            opmode=opmode,
                            finetune_tag=finetune_tag,
                            train_fraction=train_fraction,
                            log_tag=log_tag,
                    )
                        os.system(cmd)


if __name__ == "__main__":
    main()