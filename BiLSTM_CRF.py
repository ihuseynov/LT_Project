import torch
import torch.nn as nn
from utils import *
from torch.nn import init

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def log_sum(smat):
    max_score = smat.max(dim=0, keepdim=True).values
    return (smat - max_score).exp().sum(axis=0, keepdim=True).log() + max_score


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, tag_to_ix: Mapping[str, int], embedding_dim: int,
                 hidden_dim: int, char_lstm_dim=25, char_to_ix=None,
                 pre_word_embeds=None, char_embedding_dim=25,):

        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim

        print(f"out_channels: {char_lstm_dim}, hidden_dim: {hidden_dim}, ")

        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            torch.nn.init.xavier_uniform_(self.char_embeds.weight)
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2, 0),)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(embedding_dim + self.out_channels, hidden_dim, bidirectional=True)

        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

        self.hidden2tag = nn.Linear(hidden_dim * 2, self.tagset_size)
        init.xavier_normal_(self.hidden2tag.weight.data)
        init.normal_(self.hidden2tag.bias.data)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[:, tag_to_ix[START_TAG]] = -10000
        self.transitions.data[tag_to_ix[STOP_TAG], :] = -10000

    def _get_lstm_features(self, sentence, chars, caps):

        chars_embeds = self.char_embeds(chars).unsqueeze(1)
        chars_cnn_out3 = self.char_cnn3(chars_embeds)
        chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        embeds = self.word_embeds(sentence)

        embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)  # dropout
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)  # dropout
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):

        r = torch.LongTensor(range(feats.size()[0]))
        pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_start_tags, pad_stop_tags]) + torch.sum(feats[r, tags])
        return score

    def _forward_alg(self, feats):

        alpha = torch.full((1, self.tagset_size), -10000.0)

        alpha[0][self.tag_to_ix[START_TAG]] = 0.0

        for feat in feats:
            alpha = log_sum(alpha.T + feat.unsqueeze(0) + self.transitions)
        return log_sum(alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TAG]]]).flatten()[0]

    def neg_log_likelihood(self, sentence, tags, chars, caps):

        features = self._get_lstm_features(sentence, chars, caps)

        forward_score = self._forward_alg(features)
        gold_score = self._score_sentence(features, tags)
        return forward_score - gold_score

    def forward(self, sentence, chars, caps):
        features = self._get_lstm_features(sentence, chars, caps)
        backtrace = []

        alpha = torch.full((1, self.tagset_size), -10000.0)
        alpha[0][self.tag_to_ix[START_TAG]] = 0

        for feat in features:
            smat = (alpha.T + feat.unsqueeze(0) + self.transitions)  # (tagset_size, tagset_size)
        backtrace.append(smat.argmax(0))  # column_max
        alpha = log_sum(smat)

        smat = alpha.T + 0 + self.transitions[:, [self.tag_to_ix[STOP_TAG]]]
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]

        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
        best_path.append(best_tag_id)

        return log_sum(smat).item(), best_path[::-1]
