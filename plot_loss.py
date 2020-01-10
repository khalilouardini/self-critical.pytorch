import pickle as pkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rl', type=bool, default=False)
args = parser.parse_args()
print(args)

if args.rl:
    print('rl')
    file_name = 'log_attention_anderson_rl/histories_attention_anderson.pkl'
    out_file = 'figures/rl'
else:
    print('no rl')
    file_name = 'log_attention_anderson/histories_attention_anderson.pkl'
    out_file = 'figures/no_rl'

file = open(file_name, 'rb')
data = pkl.load(file)

print(os.getcwd())
if not os.path.exists(out_file):
    os.makedirs(out_file)


losses = {}
for key in data.keys():
    losses[key] = {'iterations':[], 'losses':[]}

val_loss_key = data.keys()[0]
for key in data.keys()[1:]:
    
    for iter_n in data[key].keys():
        losses[key]['iterations'].append(iter_n)
        losses[key]['losses'].append(data[key][iter_n])

    # sorting according to iterations
    list1 = losses[key]['iterations']
    list2 = losses[key]['losses']
    list1, list2 = zip(*sorted(zip(list1, list2)))
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))  
    losses[key]['iterations'] = list1
    losses[key]['losses'] = list2


for iter_n in data[val_loss_key].keys():
    losses[val_loss_key]['iterations'].append(iter_n)
    losses[val_loss_key]['losses'].append(data[val_loss_key][iter_n]['loss'])
list1 = losses[val_loss_key]['iterations']
list2 = losses[val_loss_key]['losses']
list1, list2 = zip(*sorted(zip(list1, list2)))
list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
losses[val_loss_key]['iterations'] = list1
losses[val_loss_key]['losses'] = list2

    
for key, item in losses.items():
    plt.figure()
    plt.title(key)
    plt.xlabel('iterations')
    plt.ylabel('losses')
    X = item['iterations']
    Y = item['losses']
    plt.plot(X, Y)
    plt.savefig(os.path.join(out_file,key)+'.png')
print('done')


print(data[val_loss_key][iter_n]['predictions'][:10])