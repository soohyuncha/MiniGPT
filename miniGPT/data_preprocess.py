from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch

import random

class DataPreprocessor :
    def __init__(self, batch_size, chunk_size) :
        # Configuration
        self.tokenizer = get_tokenizer('basic_english')
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Download dataset
        # WikiText2 from torchtext
        self.train_ds, self.test_ds = WikiText2('./data', split=('train', 'test'))
 
        # Build vocab
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, self.train_ds), specials=['<unk>', '<pad>'])
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.idx_to_token = self.vocab.get_itos()   # list
        self.token_to_idx = self.vocab.get_stoi()   # dict; {token: idx}
        
        # Data preprocessing
        # 1) Tokenize and convert into sequence of index. Also append special <pad> token
        # 'I am a student' -> ['i', 'am', 'a', 'student', '<pad'>] -> [3152, 14, 20, 59, 1]
        self.train_ds = self._preprocess_raw_sentence(self.train_ds)
        self.test_ds = self._preprocess_raw_sentence(self.test_ds)
        
        # 2) Shuffle ordering
        random.shuffle(self.train_ds)
        random.shuffle(self.test_ds)

        print('## Dataset Info.')
        print('\t train dataset [sentence]:', len(self.train_ds))
        print('\t test dataset [sentence]:', len(self.test_ds))
        
        # 3) Convert into list of chunk
        #    Size: (n, chunk_size + 1)
        self.train_ds = self._split_chunk(self.train_ds)
        self.test_ds = self._split_chunk(self.test_ds)
        
        print('\t train dataset [chunk]:', len(self.train_ds))
        print('\t test dataset [chunk]:', len(self.test_ds))

        # 4) Make it into DataLoader
        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

        print('\t train dataset [batch]:', len(self.train_loader))
        print('\t test dataset [batch]:', len(self.test_loader))
        print('\n')

    # tokenize & convert to sequence of index
    def _preprocess_raw_sentence(self, dataset) :
        dataset_preprocessed = []
        for x in dataset:
            # 1) tokenize
            x = self.tokenizer(x)
            # 2) append <pad>
            x.append('<pad>')
            # 3) convert into index
            dataset_preprocessed.append(self.vocab(x))
        return dataset_preprocessed

    # split dataset into fixed-size chunk
    # list of random length sentence -> list of fixed length chunk
    def _split_chunk(self, dataset):
        dataset_chunk = []
        cnt = 0
        chunk = []
        for sentence in dataset:
            for word in sentence:
                chunk.append(word)
                cnt += 1
                if cnt == self.chunk_size + 1 :
                    dataset_chunk.append(chunk)
                    chunk = []
                    cnt = 0
        return torch.tensor(dataset_chunk)


    def show_train_loader_sample(self, sample_num):
        for idx, data_batch in enumerate(self.train_loader) :
            for i, data in enumerate(data_batch):
                token_seq = ' '.join([self.idx_to_token[x] for x in data])
                print(i+1, 'th sequence of chunk:', token_seq, '\n')
            if idx == sample_num-1:
                break

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_vocab_size(self):
        return len(self.vocab)

    def get_token_to_idx(self):
        return self.token_to_idx

    def get_idx_to_token(self):
        return self.idx_to_token

    def get_vocab(self):
        return self.vocab

    def get_tokenizer(self):
        return self.tokenizer
    
if __name__ == '__main__' :
    DP = DataPreprocessor(batch_size=8, chunk_size=100)
    DP.show_train_loader_sample(1)

