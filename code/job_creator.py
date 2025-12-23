
import os

CMD = "bsub -q short -oo ./job_logs/bootstrap_{substrate}_{opmode}_{finetune_tag}_{test_fraction}.log -R \"rusage[mem=4GB]\" \"source .venv/bin/activate; python3 code/bootstrap.py --substrate {substrate} --opmode {opmode} --{finetune_tag} --test_fraction {test_fraction} --n_bootstrap 1000\""
SUBSTRATES = [
    "naphthyl_acetate",
    "p-nitrophenyl_acetate",
    "p-nitrophenyl_octanoate",
    "malathion",
    "paraoxon",
    "nonanoic_lactone",]
OPMODES = ["mean", "flat"]
FINETUNE_TAGS = ["finetune", "no-finetune"]
TEST_FRACTIONS = [0.7, 0.5, 0.3]


def main():
    for substrate in SUBSTRATES:
        for opmode in OPMODES:
            for finetune_tag in FINETUNE_TAGS:
                for test_fraction in TEST_FRACTIONS:
                    cmd = CMD.format(
                        substrate=substrate,
                        opmode=opmode,
                        finetune_tag=finetune_tag,
                        test_fraction=test_fraction,
                    )
                    os.system(cmd)


if __name__ == "__main__":
    main()