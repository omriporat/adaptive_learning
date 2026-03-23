import argparse
from itertools import permutations
from functools import reduce


def parse_args():
    desc = "Calculate the number of permutations of a resfile for a given minimum and maximum number of mutations."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("resfile", help="Path to the resfile.")
    parser.add_argument("min", type=int, help="Minimum number of mutations.")
    parser.add_argument("max", type=int, help="Maximum number of mutations.")
    args = parser.parse_args()
    return args


def parse_resfile(resfile_path):
    with open(resfile_path, "r") as f:
        lines = f.readlines()

    # skip the first two lines
    lines = lines[2:]
    mutations_dict = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid line in resfile: {line}")
        
        mutations_dict[parts[0]] = len(parts[3])

    return mutations_dict


def calculate_permutations(mutations_dict, min_mutations, max_mutations):
    total_permutations = 0
    possible_mut_nums = list(mutations_dict.values())
    possible_mut_nums = [m - 1 for m in possible_mut_nums]  # Subtract 1 to exclude the native residue

    n = len(possible_mut_nums)
    # dp[k] will store the number of ways to have exactly k mutations
    dp = [0] * (n + 1)
    dp[0] = 1  # Base case: 1 way to have 0 mutations (native)

    for m_i in possible_mut_nums:
        # We iterate backwards to avoid using the same position's 
        # mutation multiple times in the same step
        for k in range(n, -1, -1):
            # The new ways to get k mutations is:
            # (Ways to have k mutations already) + (Ways to have k-1 mutations * available mutants at this site)
            if k > 0:
                dp[k] = dp[k] + (dp[k-1] * m_i)
            else:
                # dp[0] remains 1 (the native sequence)
                pass
                
    # Sum the permutations within the specified range
    total_permutations = sum(dp[min_mutations : max_mutations + 1])
    return total_permutations, dp


def main():
    args = parse_args()
    mutations_dict = parse_resfile(args.resfile)
    total_permutations = calculate_permutations(mutations_dict, args.min, args.max)
    print(f"Total permutations: {total_permutations}")
        

if __name__ == "__main__":
    main()
