from data_preprocess import *
from model_spatten import *
#from model import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', type=str, help='Train new model or load pretrained (train/load)', default='load')
    parser.add_argument('-epochs', '--epochs', type=int, help='Epochs of training', default=5)

    args = parser.parse_args()

    batch_size = 8
    chunk_size = 200

    d_model = 768
    d_hidden = 768 * 2
    n_layer = 12
    n_head = 12

    """d_model = 128
    d_hidden = 512
    n_layer = 1
    n_head = 4"""


    #model_path = './model/miniGPT_small.pt'
    model_path = './model/miniGPT_spatten_full_dim.pt'
    myDataPreprocessor = DataPreprocessor(batch_size=batch_size, chunk_size=chunk_size)
    vocab_size = myDataPreprocessor.get_vocab_size()
    train_loader = myDataPreprocessor.get_train_loader()
    test_loader = myDataPreprocessor.get_test_loader()
    idx_to_token = myDataPreprocessor.get_idx_to_token()
    vocab = myDataPreprocessor.get_vocab()
    tokenizer = myDataPreprocessor.get_tokenizer()

    # check if cuda is available
    if (torch.cuda.is_available()):
        device = 'cuda'
    else :
        device = 'cpu'
    #device = 'cpu'
    
    print('> Device:', device, '\n')

    myGPT = miniGPT(vocab_size=vocab_size, max_len=chunk_size, d_model=d_model, d_hidden=d_hidden, n_layer=n_layer, n_head=n_head,
            device=device)
    
    myGPT.to(device)

    # Train a new model or load a pre-trained model
    if args.mode == 'train':
        epochs = args.epochs
        train(myGPT, train_loader, epochs=epochs, log_interval=100, device=device)
        torch.save(myGPT.state_dict(), model_path)

    elif args.mode == 'load' :
        myGPT.load_state_dict(torch.load(model_path))

    # Evaluation
    #evaluate_loss(myGPT, train_loader, device=device)
    #evaluate_loss(myGPT, test_loader, device=device)

    generate_test_sentence(myGPT, test_loader, tokenizer, vocab, n_batch=3, chunk_size=chunk_size, device=device)

    # Generate with prompt
    while (1):
        prompt = input('> Prompt: ')
        prompt = vocab(tokenizer(prompt))
        max_gen_len = chunk_size - len(prompt)
        gen_seq = generate(myGPT, prompt, max_gen_len=max_gen_len, vocab=vocab, attn_print=False, device=device)
        print('> Generated sequence:', gen_seq)
    
