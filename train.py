import itertools
import sys
import torch
import pickle
from tqdm import tqdm
from torch.autograd import Variable
from utils import *
from BiLSTM_CRF import BiLSTM_CRF
import numpy as np

pre_emb = "data/glove.6B.100d.txt"
name = "test"
mapping_file = "models/mapping.pkl"
models_path = "models/"
model_name = models_path + name
tmp_model = model_name + ".tmp"


def evaluating(model, datas, best_F):

    prediction = []
    save = True
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars = data['chars']
        caps = data['caps']

        len_chars = [len(c) for c in chars]
        max_char = max(len_chars)
        mask_char = np.zeros((len(len_chars), max_char), dtype='int')
        for i, c in enumerate(chars):
            mask_char[i, :len_chars[i]] = c

        mask_char = Variable(torch.LongTensor(mask_char))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))

        val, out = model(dwords, mask_char, dcaps)
        predicted_id = out

        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in open(scoref, 'r', encoding='utf8')]

    for i, line in enumerate(eval_lines):
        # print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                # print('the best F is ', new_F)

    # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #     "ID", "NE", "Total",
    #     *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    # ))
    # for i in range(confusion_matrix.size(0)):
    #     print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #         str(i), id_to_tag[i], str(confusion_matrix[i].sum().item()),
    #         *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
    #           ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
    #     ))
    # print()
    return best_F, new_F, save


def train():
    learning_rate = 0.015
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    losses = []
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    best_train_F = -1.0
    eval_every = 200
    count = 0
    sys.stdout.flush()

    model.train(True)

    for iter, index in enumerate(tqdm(np.random.permutation(len(train_data)))):
        data = train_data[index]
        model.zero_grad()
        count += 1
        sentence_in = data["words"]
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data["tags"]
        chars = data["chars"]

        # char lstm

        # char cnn
        chars_length = [len(c) for c in chars]
        char_maxl = max(chars_length)
        chars_mask = np.zeros((len(chars_length), char_maxl), dtype="int")
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c
        chars_mask = Variable(torch.LongTensor(chars_mask))

        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data["caps"]))

        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars_mask, caps)
        loss += neg_log_likelihood.data.item() / len(data["words"])
        neg_log_likelihood.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (count % eval_every == 0 and count > (eval_every * 20) or count % (eval_every * 4) == 0 and count <
                (eval_every * 20)):
            model.train(False)
            best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
            best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
            if save:
                torch.save(model, model_name)

            best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
            sys.stdout.flush()
            # print(best_test_F)

            model.train(True)

        if count % len(train_data) == 0:
            lr = learning_rate / (1 + 0.05 * count / len(train_data))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


if __name__ == "__main__":

    lower = 1
    zeros = 0
    tag_scheme = "iobes"
    word_dim = 100
    reload = 0

    train_file = "data/eng.train"
    dev_file = "data/eng.testa"
    test_file = "data/eng.testb"
    test_train_file = "data/eng.train50000"

    train_sentences = load_sentences(train_file, zeros)
    dev_sentences = load_sentences(dev_file, zeros)
    test_sentences = load_sentences(test_file, zeros)
    test_train_sentences = load_sentences(test_train_file, zeros)

    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)
    update_tag_scheme(test_train_sentences, tag_scheme)

    dico_words_train = mapping("word", train_sentences, lower)[0]

    dico_words, word_to_id, id_to_word = augment_with_pretrained(dico_words_train.copy(), pre_emb,
                                                                 list(itertools.chain.from_iterable(
                                                                [[w[0] for w in s] for s in
                                                                dev_sentences + test_sentences])) if not
                                                                 1 else None,)

    dico_chars, char_to_id, id_to_char = mapping("char", train_sentences)
    dico_tags, tag_to_id, id_to_tag = mapping("tag", train_sentences)

    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
    dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower)
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)
    test_train_data = prepare_dataset(test_train_sentences, word_to_id, char_to_id, tag_to_id, lower)

    print(f"{len(train_data)} / {len(dev_data)} / {len(test_data)} sentences in train / dev / test.")

    all_word_embeds = {}
    for i, line in enumerate(open(pre_emb, "r", encoding="utf-8")):
        s = line.strip().split()
        if len(s) == word_dim + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print(f"Loaded {len(all_word_embeds)} pretrained embeddings.")

    with open(mapping_file, "wb") as f:

        mappings = {"word_to_id": word_to_id, "tag_to_id": tag_to_id, "char_to_id": char_to_id, "word_embeds": word_embeds, }
        pickle.dump(mappings, f)

    print("word_to_id: ", len(word_to_id))

    model = BiLSTM_CRF(vocab_size=len(word_to_id), tag_to_ix=tag_to_id,
                       embedding_dim=word_dim,
                       hidden_dim=200,
                       char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,)

    if reload:
        model = torch.load(model_name)

    train()
