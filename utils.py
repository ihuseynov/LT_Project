import os
import re
import BiLSTM_CRF
from collections import Counter

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def create_mapping(dico):

    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def iob_to_iobes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


def load_sentences(path, zeros):
    sentences = []
    sentence = []
    for line in open(path, 'r', encoding='utf-8'):
        line = re.sub("\d", "0", line.rstrip()) if zeros else line.rstrip()

        if not line and len(sentence) > 0:

            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
            sentence = []

        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)

    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def iob2(tags):
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            return False
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return True


def update_tag_scheme(sentences, tag_scheme):

    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]

        if not iob2(tags):
            continue

        elif tag_scheme == 'iobes':
            new_tags = iob_to_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True):

    def f(x): return x.lower() if lower else x
    data = []
    for sentence in sentences:
        str_words = [w[0] for w in sentence]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]
                 for w in str_words]
        caps = []

        for s in str_words:
            if s.lower() == s:
                caps.append(0)
            elif s.upper() == s:
                caps.append(1)
            elif s[0].upper() == s[0]:
                caps.append(2)
            else:
                caps.append(3)

    tags = [tag_to_id[w[-1]] for w in sentence]

    data.append({'str_words': str_words, 'words': words, 'chars': chars, 'caps': caps, 'tags': tags, })
    return data


def mapping(type_mapping, sentences, lower=None):

    if type_mapping == "tag":
        tags = [word[-1] for s in sentences for word in s]
        dico = dict(Counter(tags))
        dico[BiLSTM_CRF.START_TAG] = -1
        dico[BiLSTM_CRF.STOP_TAG] = -2
        tag_to_id, id_to_tag = create_mapping(dico)
        print(f"Found {len(dico)} unique named entity tags")
        return dico, tag_to_id, id_to_tag

    elif type_mapping == "char":
        chars = ''.join([w[0] for s in sentences for w in s])
        dico = dict(Counter(chars))
        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000

        char_to_id, id_to_char = create_mapping(dico)
        print(f"Found {len(dico)} unique characters")
        return dico, char_to_id, id_to_char

    elif type_mapping == "word":
        words = [x[0].lower() if lower else x[0] for s in sentences for x in s]
        dico = dict(Counter(words))
        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000
        dico = {k: v for k, v in dico.items() if v >= 3}
        word_to_id, id_to_word = create_mapping(dico)

        print(f"Found {len(dico)} unique words ({len(words)} in total)")
        return dico, word_to_id, id_to_word


def augment_with_pretrained(dictionary, ext_emb_path, words):
    print(f'Loading pretrained embeddings from {ext_emb_path}...')
    assert os.path.isfile(ext_emb_path)

    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in open(ext_emb_path, 'r', encoding='utf-8')
    ])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


