import sys
import numpy as np

def read_alpha_labels(alpha_label_file):
    probs = []
    with open(alpha_label_file, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        prob1, prob2 = line.strip().split('\t')
    probs.append([float(prob1), float(prob2)])

    return probs


def calc_entropy(probs):
    if type(probs) is list:
        probs = np.array(probs)

    log_probs = np.log(probs)
    entropy = - np.sum(probs * log_probs)
    return entropy



#TODO currently, this script is only for alphaNLI


assert(len(sys.argv) == 2)
pred_file = sys.argv[1]


alpha_probs = read_alpha_labels(pred_file)

sum_entropy = 0

for probs in alpha_probs:
    entropy = calc_entropy(probs)
    sum_entropy += entropy


print(f"Mean entropy: {sum_entropy / len(probs)}")



