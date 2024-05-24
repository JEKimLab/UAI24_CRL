import pandas as pd
import torch.utils.data
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

classes = {
    'astro-ph': 0, 'cond-mat': 1, 'cs': 2, 'eess': 3, 'hep-ph': 4,
    'hep-th': 5, 'math': 6, 'physics': 7, 'quant-ph': 8, 'stat': 9
}


class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()
        class_str2idx = {}
        class_idx2str = []
        texts, labels = [], []

        with open(data_path) as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                text = ""
                for tx in line[1:len(line) - 1]:
                    text += tx.lower()
                    text += " "
                # print(text)
                if not line[-1] in class_str2idx.keys():
                    class_str2idx[line[-1]] = len(class_idx2str)
                    class_idx2str.append(line[-1])
                # print(class_str2idx)
                label = class_str2idx[line[-1]]
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(
            filepath_or_buffer=dict_path, header=None, sep=" ",
            quoting=csv.QUOTE_NONE, usecols=[0]
        ).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        print(class_str2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1
        return document_encode.astype(np.int64), label


if __name__ == '__main__':
    # import nltk
    # nltk.download('punkt')
    test = MyDataset(
        data_path='/home/xfang23/dataset/ArXiv-10/arxiv100.csv',
        dict_path='/home/xfang23/dataset/ArXiv-10/glove/glove.6B.50d.txt',
        max_length_sentences=10, max_length_word=40
    )
    print(len(test))
    print(test.__getitem__(index=1)[0].shape)
    dl = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, num_workers=2)

    data = []
    label = []
    for idx, (x, t) in enumerate(dl):
        print(x.size(), t.size())
        # print(x.numpy().shape, t.numpy().shape)
        data.append(x.numpy())
        label.append(t.numpy())
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)
    print(data.shape, label.shape)
    # Save both arrays into a single file
    np.savez('/home/xfang23/dataset/ArXiv-10/data-50d.npz', data=data, label=label)
