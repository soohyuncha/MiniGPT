from data_preprocess import *
from model import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, help='Dataset name (WikiText2/Wikitext103)', default='WikiText2')
    parser.add_argument('--dataset_mode', type=str, help='Preprocess dataset or load preprocessed (preprocess/load)', default='preprocess')
    parser.add_argument('--train_mode', type=str, help='Train new model or load pretrained (train/load)', default='load')
    parser.add_argument('--train_epochs', type=int, help='Epochs of training', default=5)
    parser.add_argument('--gen_mode', type=str, help='Generation mode(no/input/eval_set)', default='eval_set')
    parser.add_argument('--model_path', type=str, help='Path for model(save or load)', default='./model/miniGPT.pt')

    parser.add_argument('--batch_size', type=int, help='Batch size of dataset', default=8)
    parser.add_argument('--chunk_size', type=int, help='Chunk size of sentence', default=200)

    parser.add_argument('--d_model', type=int, help='Embedding dimension', default=768)
    parser.add_argument('--d_hidden', type=int, help='Hidden dimension of ffn layer', default=768*2)
    parser.add_argument('--n_layer', type=int, help='Number of decoders', default=12)
    parser.add_argument('--n_head', type=int, help='Numbder of heads', default=12)

    
    args = parser.parse_args()

    # Model configurations
    model_path = args.model_path

    batch_size = args.batch_size
    chunk_size = args.chunk_size

    d_model = args.d_model
    d_hidden = args.d_hidden
    n_layer = args.n_layer
    n_head = args.n_head

    print('> Configuration set done\n')
    
    # Data Preprocess
    myDataPreprocessor = DataPreprocessor(batch_size=batch_size,
            chunk_size=chunk_size,
            dataset_name=args.dataset_name,
            is_load=args.dataset_mode)

    vocab_size = myDataPreprocessor.get_vocab_size()
    train_loader = myDataPreprocessor.get_train_loader()
    test_loader = myDataPreprocessor.get_test_loader()
    idx_to_token = myDataPreprocessor.get_idx_to_token()
    vocab = myDataPreprocessor.get_vocab()
    tokenizer = myDataPreprocessor.get_tokenizer()

    print('> Data preprocess done\n')

    # Check if cuda is available
    if (torch.cuda.is_available()):
        device = 'cuda'
    else :
        device = 'cpu'
    #device = 'cpu'
    
    print('> Device:', device, '\n')

    # Model
    myGPT = miniGPT(vocab_size=vocab_size, max_len=chunk_size, d_model=d_model, d_hidden=d_hidden, n_layer=n_layer, n_head=n_head,
            device=device)
    myGPT.to(device)

    # Train a new model or load a pre-trained model
    if args.train_mode == 'train':
        epochs = args.train_epochs
        train(myGPT, train_loader, epochs=epochs, log_interval=100, device=device)
        torch.save(myGPT.state_dict(), model_path)

    elif args.train_mode == 'load' :
        myGPT.load_state_dict(torch.load(model_path))

    else :
        print('\n\n**************** Error *********************')
        print("> Can't load model")
        exit()


    # Evaluate model; loss and perplexity
    evaluate(myGPT, train_loader, device=device)
    evaluate(myGPT, test_loader, device=device)


    # Generate some sample sentences
    if args.gen_mode == 'input':
        while (1):
            prompt = input("> Enter your prompt (Type 'exit' if you want to stop generating): ")
            if prompt == 'exit' :
                break
            max_gen_len = chunk_size - len(prompt)
            gen_seq = generate(myGPT, prompt, max_gen_len=max_gen_len, vocab=vocab, tokenizer=tokenizer, device=device)
            print('> Generated sequence:', gen_seq)

    elif args.gen_mode == 'eval_set':
        generate_test_sentence(myGPT, test_loader, tokenizer, vocab, n_batch=3, chunk_size=chunk_size, device=device)


        
