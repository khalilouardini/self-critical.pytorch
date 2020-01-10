import json
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--rl', type=bool, default=False)
args = parser.parse_args()

if args.rl:
    print('rl')
    file_name = 'log_attention_anderson_rl/histories_attention_anderson.pkl'
else:
    print('no rl')
    file_name = 'log_attention_anderson/histories_attention_anderson.pkl'
file = open(file_name, 'rb')
predictions = pkl.load(file)['val_result_history']

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

import nltk

scores_bleu = {}
for iter_n in predictions:
    scores[iter_n] = 0
    predictions_i = predictions[iter_n]['predictions']
    for pred_i in predictions_i:
        image_id = pred_i['image_id']
        caption = pred_i['caption']

        reference_caption = captions_dict[image_id]
        
        scores_bleu[iter_n] += nltk.translate.bleu_score.sentence_bleu(reference_caption, caption)
        
    
print(scores_bleu)
