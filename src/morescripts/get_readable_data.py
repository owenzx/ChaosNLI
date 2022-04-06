import json
import sys

def read_jsonl(f):
    result = []
    with open(f, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        result.append(json.loads(line))
    return result




def write_example_friendly(examples, f):
    with open(f, 'w') as fw:
        for ex in examples:
            print(f"ID:\t{ex['id']}", file=fw)
            print(f"SEN1:\t{ex['sentence1']}", file=fw)
            print(f"SEN2:\t{ex['sentence2']}", file=fw)
            if 'hyp1' in ex.keys():
                print(f"HYP1:\t{ex['hyp1']}", file=fw)
                print(f"HYP2:\t{ex['hyp2']}", file=fw)
            if len(ex['label_dist']) == 3:
                print(f"Entailment: {ex['label_dist'][0]}, Neutral: {ex['label_dist'][1]}, Contradiction: {ex['label_dist'][2]}", file=fw)
            elif len(ex['label_dist']) == 2:
                print(f"hyp1: {ex['label_dist'][0]}, hyp2: {ex['label_dist'][1]}", file=fw)
            else:
                raise NotImplementedError
            print(f"REASON: ", file=fw)
            print(f"\n\n", file=fw)





chaos_file = sys.argv[1]
ogn_file = sys.argv[2]
out_file = sys.argv[3]


chaos_data = read_jsonl(chaos_file)
ogn_data = read_jsonl(ogn_file)

readable_example = []

if 'mnli' in chaos_file or 'snli' in chaos_file:
    ogn_dict = {}
    for ex in ogn_data:
        ogn_dict[ex['pairID']] = ex

    for ex in chaos_data:
        uid = ex['uid']
        ogn_ex = ogn_dict[uid]

        readable_example.append({'sentence1': ogn_ex['sentence1'],
                                 'sentence2': ogn_ex['sentence2'],
                                 'id': uid,
                                 'label_dist': ex['label_dist']
                                 })

elif 'alpha' in chaos_file:
    for i, ex in enumerate(chaos_data):
        uid = i
        readable_example.append({'sentence1': ex['example']['obs1'],
                                 'sentence2': ex['example']['obs2'],
                                 'hyp1': ex['example']['hyp1'],
                                 'hyp2': ex['example']['hyp2'],
                                 'id': uid,
                                 'label_dist': ex['label_dist']
                                 })

else:
    raise NotImplementedError


write_example_friendly(readable_example, out_file)

