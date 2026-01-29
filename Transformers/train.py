import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader, random_split
from config import get_config, get_weights_file_path, latest_weights_file_path

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer # Training the tokenizer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
import warnings
from pathlib import Path

def get_all_sentences(ds, lang):
    """
        This methos load all the sentences of language corrosponding to a tokenizer.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
        config: dictonary of configurations
        ds: dataset
        lang: Language of encoder or decoder
    """
    # config['tokenizer_file'] = "../tokenizers/tokenizer_{0}.json"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # If the frequency of those words are >= 2 then only they are the part of vocabulary
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer) # The method gives all the senteneces from dataset
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Loading Dataset from huggingface
    ds_raw = load_dataset(path='Helsinki-NLP/opus_books', name=f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, lang=config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, lang=config['lang_tgt'])

    # Splitting the dataset (90-10)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Creating the dataset objects
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    # Finding the max length of sentence in each input and output
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    # Creating dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], config["d_model"], N = 1, h = 4
    )
    # NOTE: DID CHANGES
    """
        he function creates the model but doesn't return it!
    """
    return model

def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # NOTE: DID CHANGES
    """
        Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    """
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # Loading dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Loading model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    # Defining loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1
    ).to(device)


    # Model Training
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)
            # (batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (bathc, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # NOTE: DID CHANGES
            """
                # ❌ Wrong - this creates a SET with one string, not a dictionary
                batch_iterator.set_postfix({f"loss: {loss.item():6.3f}"})

                # ✅ Correct - this creates a DICTIONARY with key-value pair
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            """
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss on tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropogation
            loss.backward()

            # Update the weight
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)