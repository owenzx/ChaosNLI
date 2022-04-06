import torch
import sys
import numpy as np
import json
from torch.nn import functional as F
import numpy as np




def read_jsonl(path):
    result = []
    with open(path, 'r') as fr:
        for line in fr.readlines():
            result.append(json.loads(line))
    return result


def write_jsonl(path, dict_lst):
    with open(path, 'w') as fw:
        for d in dict_lst:
            json.dump(d, fw)
            fw.write('\n')





def T_scaling(logits, args):
  temperature = args.get('temperature', None)
  return torch.div(logits, temperature)

def T_scaling_lst(logits, temperature):
    if type(logits) is list:
        logits = np.asarray(logits)

    logits =logits / temperature
    return logits.tolist()



def get_JSD(pred_logits, gt_probs):
    pred_probs = F.softmax(pred_logits, -1)
    # pred_log_probs = F.log_softmax(pred_logits, -1)
    # gt_logprobs = torch.log(gt_probs+1e-15)
    m = (pred_probs + gt_probs) / 2
    log_m = torch.log(m)

    # print('PRED:')
    # print(pred_probs[0])
    # print(gt_probs[0])
    sum_kldiv =  F.kl_div(log_m, pred_probs, reduction='none').sum(-1) + F.kl_div(log_m, gt_probs, reduction='none').sum(-1)
    sum_kldiv = torch.max(sum_kldiv, torch.Tensor([0.]).to(sum_kldiv.device))
    # print(sum_kldiv.min())
    JSD = torch.sqrt(sum_kldiv / 2)

    JSD = JSD.mean()
    # print("JSD:")
    # print(JSD)
    assert(not torch.isnan(JSD))
    return JSD




logits_list = []
labels_list = []
temps = []
losses = []

# read logits and labels
assert(len(sys.argv) in [4,5])
inp_path = sys.argv[1]
label_path = sys.argv[2]

out_path = sys.argv[3]
if len(sys.argv) == 5:
    final_temp = float(sys.argv[4])
else:


    inp_dict = read_jsonl(inp_path)
    label_dict = read_jsonl(label_path)
    for label_info in label_dict:
        uid = label_info['uid']
        if 'logits' in inp_dict[0]['roberta-large'][uid]:
            logits = inp_dict[0]['roberta-large'][uid]['logits']
        else:
            probs = inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
            logits = np.log(probs).tolist()
            inp_dict[0]['roberta-large'][uid]['logits'] = logits
            del inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
        label_dist = label_info['label_dist']
        logits_list.append(torch.FloatTensor(logits))
        labels_list.append(torch.FloatTensor(label_dist))




    # calibration
    temperature = torch.nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
    #criterion = torch.nn.KLDivLoss(reduction='batchmean')
    criterion = get_JSD


    optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    # Create tensors
    logits_list = torch.stack(logits_list,0).cuda()
    labels_list = torch.stack(labels_list,0).cuda()

    def _eval():
      loss = criterion(T_scaling(logits_list, args), labels_list)
      loss.backward()
      temps.append(temperature.item())
      losses.append(loss)
      return loss


    optimizer.step(_eval)

    # print(temps)
    # print(temperature.item())

    final_temp = temperature.item()
print(final_temp)

for uid in inp_dict[0]['roberta-large'].keys():
    if 'logits' not in inp_dict[0]['roberta-large'][uid]:
        probs = inp_dict[0]['roberta-large'][uid]['predicted_probabilities']
        logits = np.log(probs).tolist()
        inp_dict[0]['roberta-large'][uid]['logits'] = logits
        del inp_dict[0]['roberta-large'][uid]['predicted_probabilities']

    inp_dict[0]['roberta-large'][uid]['logits'] = T_scaling_lst(inp_dict[0]['roberta-large'][uid]['logits'], final_temp)

write_jsonl(out_path, inp_dict)


