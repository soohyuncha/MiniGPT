import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import time

class PositionEncodingLayer(nn.Module) :
    def __init__(self, max_len, d_model):
        super().__init__()
        # create position encoding table of shape (max_len, d_model)
        pe_table = np.array([self._get_pos_angle_vec(pos, d_model) for pos in range(max_len)])
        pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])
        pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])
        pe_table = torch.tensor(pe_table, dtype=torch.float32)
        self.register_buffer('pe_table', pe_table)
        #print(self.pe_table.shape)     # (max_len, d_model)

    def _get_pos_angle_vec(self, pos, d_model) :
        return np.array([pos / np.power(10000, (2*(i//2)) / d_model) for i in range(d_model)])
    
    def forward(self, x, start, end) :
        # input x: (batch_size, seq_len, d_model)
        # pe_table: (max_len, d_model)
        # output: (batch_size, seq_len, d_model)
        x = x + self.pe_table[start:end].clone().detach()
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # q, k, v: (batch_size, n_head, seq_len, head_dim)
        # k^T: (batch_size, n_head, head_dim, seq_len)
        k_trans = k.transpose(-1, -2)

        # attn_prob: (batch_size, n_head, seq_len, seq_len)
        attn_prob = torch.matmul(q, k_trans) / (self.d_model ** 0.5)

        # mask: (seq_len, seq_len), upper triangle except for diagonal is True
        # Ex. (3, 3)
        # F T T
        # F F T
        # F F F
        seq_len = attn_prob.size(2)
        causal_mask = (torch.tril(torch.ones(seq_len, seq_len)) == 0).to(self.device)
        # after masking
        # Ex. (3, 3)
        # num -inf -inf
        # num  num -inf
        # num  num  num
        attn_prob.masked_fill_(causal_mask, -np.inf)
        attn_prob = F.softmax(attn_prob, dim=-1)
        attn_prob = self.dropout_layer(attn_prob)

        attn_out = torch.matmul(attn_prob, v)
        return attn_out

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_hidden)
        x_hidden = self.fc1(x)
        x_hidden = F.relu(x_hidden)
    
        # (batch_size, seq_len, d_hidden) -> (batch_size, seq_len, d_model)
        out = self.fc2(x_hidden)
        out = self.dropout_layer(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, device):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)

        # Dimension check
        # Instead of d_model, should it be head_dim?
        self.attention = ScaledDotProductAttention(d_model, dropout, device)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def _split_head(self, x, n_head, head_dim):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_head, head_dim)
        new_shape = x.size()[:-1] + (n_head, head_dim)
        x = x.view(new_shape)
        # (batch_size, seq_len, n_head, head_dim) -> (batch_size, n_head, seq_len, head_dim)
        x = x.transpose(1, 2)
        return x

    def _merge_head(self, x, n_head, head_dim):
        # (batch_size, n_head, seq_len, head_dim) -> (batch_size, seq_len, n_head, head_dim)
        x = x.transpose(2, 1).contiguous()
        # (batch_size, seq_len, n_head, head_dim) -> (batch_size, seq_len, n_head*head_dim)
        new_shape = x.size()[:-2] + (n_head * head_dim,)
        x = x.view(new_shape)
        return x

    def forward(self, x, layer_past=None):
        # Generate 3 matrix: Q (query), K (key), V (value)
        # (batch_size, seq_len, d_model)
        q = self.fc_query(x)
        k = self.fc_key(x)
        v = self.fc_value(x)
        
        # Split into multi-head
        # (batch_size, n_head, seq_len, head_dim)
        q = self._split_head(q, self.n_head, self.head_dim)
        k = self._split_head(k, self.n_head, self.head_dim)
        v = self._split_head(v, self.n_head, self.head_dim)

        # If there exists cached K, V
        if layer_past != None:
            # past_k, past_v: (batch_size, n_head, n, head_dim)
            past_k, past_v = layer_past[0], layer_past[1]
            # concatenate; (batch_size, n_head, n+1, head_dim)
            k, v = torch.cat((past_k, k), dim=-2), torch.cat((past_v, v), dim=-2)
        # Caching K, V
        present = torch.stack((k, v))

        # attn_out: (batch_size, n_head, seq_len, head_dim)
        attn_out = self.attention(q, k, v)

        # Merge multi-head
        # (batch_size, n_head, seq_len, head_dim) -> (batch_size, seq_len, d_model)
        attn_out = self._merge_head(attn_out, self.n_head, self.head_dim)

        # (batch_size, seq_len, d_model)
        out = self.fc(attn_out)
        out = self.dropout_layer(out)
        return out, present

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, dropout, device):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_head, dropout, device)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_hidden, dropout)

    def forward(self, x, layer_past=None):
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)

        # Multi-head attention
        residual = x
        attn_x, present = self.attention(self.layer_norm_1(x), layer_past=layer_past)
        x = residual + attn_x

        # Feed forward network
        residual = x
        ffn_x = self.ffn(self.layer_norm_2(x))
        out = residual + ffn_x

        return out, present

class miniGPT(nn.Module) :
    def __init__(self, vocab_size, max_len, d_model, d_hidden, n_layer, n_head, device) :
        super().__init__()
        # configuration
        self.dropout = 0.1

        # layer definition
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.position_encoding = PositionEncodingLayer(max_len=max_len, d_model=d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.decoder_stack = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_hidden=d_hidden, n_head=n_head, dropout=self.dropout, device=device)
            for _ in range(n_layer)
        ])
        self.decoder_layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, past=None) :
        if past is None:
            past_length = 0
            past = [None] * len(self.decoder_stack)
        else:
            past_length = past[0][0].size(-2)
        # Input: (batch_size, seq_len)
        # Output: (batch_size, seq_len, d_model)
        
        # Embedding
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = self.position_encoding(x, start=past_length, end=past_length+x.size(-2))
        x = self.dropout_layer(x)

        # Decoder stack
        presents = []
        for dec_block, layer_past in zip(self.decoder_stack, past):
            # layer_past: (2, batch_size, n_head, n, head_dim)
            # present: (2, batch_size, n_head, n, head_dim)
            x, present = dec_block(x, layer_past)
            # presents: (n_layer, 2, batch_size, n_head, n, head_dim)
            presents.append(present)
           
        # Layer normalization
        # (batch_size, seq_len, d_model)
        hidden_state = self.decoder_layer_norm(x)

        # Output: (batch_size, seq_len, vocab_size)
        out = self.fc(hidden_state)
        return out, presents

def train(model, train_loader, epochs, log_interval, device='cpu') :
    # Training algorithms
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
            lr_lambda=lambda epoch: 0.95 ** epoch)

    model.train()

    batch_num = len(train_loader)
    total_loss = 0

    print('|------------------------------------------|')
    print('\t Start training', epochs, 'epochs')
    print('|------------------------------------------|')
    start_training = time.time()

    for epoch in range(epochs) :
        start = time.time()
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)

            # input_ids: (batch_size, chunk_size)
            # targets: (batch_size, chunk_size) -> (batch_size*chunk_size)
            input_ids = data[:, :-1]
            targets = data[:, 1:].contiguous().view(-1)
            
            # output: (batch_size, chunk_size, vocab_size) -> (batch_size*chunk_size, vocab_size)
            output, _ = model(input_ids)
            output = output.contiguous()
            output = output.view(output.size(0)*output.size(1), -1)
            
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx and batch_idx % log_interval == 0:
                print(f'\t Epoch {epoch+1}, Batch {batch_idx}/{batch_num}, Avg. loss {total_loss/(batch_idx+1): .3f}')
            
        end = time.time()
        print(f'> Epoch {epoch+1}/{epochs} done with {end-start: .2f} sec')
        print(f'  Avg. loss per batch: {total_loss/batch_num: .3f}')
        print('\n')
        total_loss = 0
        scheduler.step()

    end_training = time.time()
    print(f'> Training done with {end_training-start_training: .2f} sec')

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    print('|------------------------------------------|')
    print('\t Start evaluation')
    print('|------------------------------------------|')

    start = time.time()

    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            input_ids = data[:, :-1]
            targets = data[:, 1:].contiguous().view(-1)
            
            # output: (batch_size, chunk_size, vocab_size) -> (batch_size*chunk_size, vocab_size)
            output, _ = model(input_ids)
            output = output.contiguous()
            output = output.view(output.size(0)*output.size(1), -1)
            
            loss = criterion(output, targets)
            total_loss += loss.item()

    end = time.time()
    
    avg_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f'> Evaluation done with {end-start: .2f} sec')
    print(f'  Avg. loss: {avg_loss: .3f}')
    print(f'  Perplexity: {perplexity: .3f}')
    print('\n')

def generate_test_sentence(model, test_loader, tokenizer, vocab, n_batch, chunk_size, device='cpu'):
    model.eval()
    idx_to_token = vocab.get_itos()

    print('## Start generation of sentence from test dataset')
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            input_ids = data[:, :chunk_size // 2]
            
            for i in range(data.size(0)):
                prompt_seq = ' '.join([idx_to_token[x] for x in data[i, :chunk_size // 2]])
                ans_seq = ' '.join([idx_to_token[x] for x in data[i, chunk_size // 2:]])                

                # Generate with generate() function
                gen_seq = generate(model=model,
                        prompt=prompt_seq,
                        max_gen_len=chunk_size // 2,
                        vocab=vocab,
                        tokenizer=tokenizer,
                        device=device) 
                
                print(f'> Batch {batch_idx+1} th, Sample {i+1}/{data.size(0)} th')
                print('\t > Prompt:', prompt_seq, '\n')
                print('\t > Answer sentence:', ans_seq, '\n')
                print('\t > Generated sentence:', gen_seq, '\n')
                         
            if batch_idx + 1 == n_batch:
                break



def generate(model, prompt, max_gen_len, vocab, tokenizer, device='cpu'):
    # prompt: sequence of words
    # 'Hi my name is generative ...'
    model.eval()
    gen_seq_idx = []
    gen_seq_token = []

    token_to_idx = vocab.get_stoi()
    idx_to_token = vocab.get_itos()
    tokenized_prompt = vocab(tokenizer(prompt))

    with torch.no_grad():
        past = None
        for i in range(max_gen_len):
            # (seq_len) -> (1, seq_len)
            input_ids = torch.tensor(tokenized_prompt, device=device).unsqueeze(0)
        
            # caching previous K, V in 'past'
            output, past = model(input_ids, past=past)
            # convert into cpu
            # don't need to pass it to any torch library
            output = output.cpu()
            # 1th element of batch; last element of sequence;
            max_idx = output[0][-1].argmax().item()

            #prompt.append(max_idx.item())
            tokenized_prompt = [max_idx]

            gen_seq_idx.append(max_idx)
            
            # eos token
            if max_idx == token_to_idx['<pad>']:
                break

    gen_seq_token = ' '.join([idx_to_token[x] for x in gen_seq_idx])

    
    # return: string (=raw sentence)
    return gen_seq_token



if __name__ == '__main__' :
    # Test PositionEncodingLayer
    """
    myPE = PositionEncodingLayer(max_len=200, d_model=64)
    x = np.ones((8, 100, 64))
    x = torch.tensor(x)
    y = myPE(x)
    print(y.shape)
    """
    # Test dimension of MultiHeadAttention
    """
    x = np.ones((2, 5, 4))
    x[1] = 2
    x = torch.tensor(x, dtype=torch.float32)
    
    myMH = MultiHeadAttention(d_model=4, n_head=2, dropout=0.1)
    y = myMH(x)
    """

    # Test mask
    """
    mask = (torch.tril(torch.ones(5, 5)) == 0)
    #print(mask)
    a = np.ones((3, 5, 5))
    a = torch.tensor(a)
    a.masked_fill_(mask, -np.inf)
    print(a)
    """
    
