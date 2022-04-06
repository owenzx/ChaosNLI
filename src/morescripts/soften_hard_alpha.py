import sys

def read_alpha_hard_file(hard_file):
    labels = []
    with open(hard_file, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        labels.append(int(line.strip()))
    return labels

def write_alpha_dist_label(path, label_lst):
    with open(path, 'w') as fw:
        for label in label_lst:
            label = [str(x) for x in label]
            fw.write('\t'.join(label))
            fw.write('\n')


assert(len(sys.argv) == 3)
hard_file = sys.argv[1]
soft_file = sys.argv[2]

hard_labels = read_alpha_hard_file(hard_file)

soft_label_dict = {1: [0.97673324116161774, 0.02326675883838226],
                   2: [0.02326675883838226, 0.97673324116161774]}

soft_labels = [soft_label_dict[l] for l in hard_labels]

write_alpha_dist_label(soft_file, soft_labels)