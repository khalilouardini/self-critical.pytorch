import json, os
import numpy as np
import argparse
import pickle as pkl
import nltk
from rouge import Rouge
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--rl', type=bool, default=False)
args = parser.parse_args()

if args.rl:
    print('rl')
    file_name = 'log_attention_anderson_rl/histories_attention_anderson.pkl'
    out_file = 'figures/rl'
else:
    print('no rl')
    file_name = 'log_attention_anderson/histories_attention_anderson.pkl'
    out_file = 'figures/no_rl'
    
my_file = open(file_name, 'rb')
predictions = pkl.load(my_file)['val_result_history']

path_labels = 'data/f30k_captions4eval.json'

with open(path_labels, 'rb') as json_file:
    data = json_file.read()


captions_dict = {}
labels_dict = json.loads(data)["annotations"] # list of dictionaries

## creating labels dictionary in a different format: {"image_id":list of list of strs}
for D in labels_dict:
    image_id = D['image_id']
    caption = D['caption']
    if image_id not in captions_dict:
        captions_dict[image_id] = [caption.split(" ")]
    else:
        captions_dict[image_id] += [caption.split(" ")]


rouge = Rouge()

scores_bleu = {}
scores_rouge = {}
init_rouge = {'rouge-1':{'f':0, 'p':0, 'r':0},
    'rouge-2':{'f':0, 'p':0, 'r':0},
    'rouge-l':{'f':0, 'p':0, 'r':0}}

def add_rouge(red1, red2):
    for red_n in red1:
        red1[red_n]['p'] += red2[red_n]['p']
        red1[red_n]['r'] += red2[red_n]['r']
    return red1

def normalize_rouge(red, n):
    for red_n in red:
        red[red_n]['p'] /= n
        red[red_n]['r'] /= n
    return red

def compute_f(red):
    for red_n in red:
        red[red_n]['f'] = 2*red[red_n]['p']*red[red_n]['r']/(red[red_n]['p']+red[red_n]['r'])
    return red

scores_rouge_plot = [np.array([0,0,0])]*3

sorted_iterations = sorted(list(predictions.keys()))
print(sorted_iterations)

for iter_n in sorted_iterations:
    scores_bleu[iter_n] = 0
    scores_rouge[iter_n] = init_rouge
    predictions_i = predictions[iter_n]['predictions']
    for pred_i in predictions_i:
        image_id = pred_i['image_id']
        predicted_caption = pred_i['caption']
        reference_captions = captions_dict[image_id]
        # computing bleu score
        scores_bleu[iter_n] += nltk.translate.bleu_score.sentence_bleu(reference_captions, predicted_caption)
        # computing red score (must iterate over references since library does not support multiple references)
        for single_ref_cap in reference_captions:
            single_rouge = rouge.get_scores(predicted_caption, ' '.join(single_ref_cap))[0]
            scores_rouge[iter_n] = add_rouge(scores_rouge[iter_n], single_rouge)
        scores_rouge[iter_n] = normalize_rouge(scores_rouge[iter_n], len(reference_captions))
    scores_rouge[iter_n] = compute_f(scores_rouge[iter_n])
    for i, rouge_i in enumerate(scores_rouge[iter_n]):
        p, r, f = scores_rouge[iter_n][rouge_i]['p'], scores_rouge[iter_n][rouge_i]['r'], scores_rouge[iter_n][rouge_i]['f']
        scores_rouge_plot[i] = np.vstack((scores_rouge_plot[i], [p, r, f]))

print(scores_bleu)

l = scores_rouge_plot

labels = ['p', 'r', 'f']
linestyles = ['-', '-.', ':']

plt.figure(figsize=(10,10))
if args.rl:
    plt.title('with rl')
else:
    plt.title('without rl')
for i, array in enumerate(l):
    for j in range(3):
        plt.plot(sorted_iterations, array[1:,j], linestyle=linestyles[i], label=labels[j])
plt.xlabel('iterations')
plt.ylim(ymin=0, ymax=1)
plt.legend()
plt.savefig(os.path.join(out_file, 'rouge_scores.png'))
print('figure saved to: ',out_file+'/rouge_scores.png')
