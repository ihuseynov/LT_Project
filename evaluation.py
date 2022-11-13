import torch
import pickle
import numpy as np
from torch.autograd import Variable
from utils import *

mapping_file = 'models/mapping.pkl'

with open(mapping_file, 'rb') as f:
    mappings = pickle.load(f)

word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
id_to_tag = {k[1]: k[0] for k in tag_to_id.items()}
char_to_id = mappings['char_to_id']
word_embeds = mappings['word_embeds']

lower = 1
zeros = 0
tag_scheme = "iobes"

test_sentences = load_sentences("data/eng.testb", lower)
update_tag_scheme(test_sentences, tag_scheme)
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)

model = torch.load("models/test")
model_name = "models/test".split('/')[-1].split('.')[0]
model.eval()


def eval(model, all_data):
    prediction = []
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    for data in all_data:

        ground_truth_id = data['tags']
        words = data['str_words']
        chars = data['chars']
        caps = data['caps']

        len_chars = [len(c) for c in chars]
        char_maxl = max(len_chars)
        mask_chars = np.zeros((len(len_chars), char_maxl), dtype='int')

        for i, c in enumerate(chars):
            mask_chars[i, :len_chars[i]] = c
        mask_chars = Variable(torch.LongTensor(mask_chars))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))

        val, out = model(dwords, mask_chars, dcaps)

        predicted_id = out

        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1

        prediction.append('')

    predf = eval_temp + '/pred.' + model_name
    scoref = eval_temp + '/score.' + model_name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    with open(scoref, 'r') as f:
        for l in f.readlines():
            print(l.strip())

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])))

    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum().item()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])))


eval(model, test_data)
