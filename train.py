
import torch
import torch.nn as nn
from torch.utils.data import random_split,Dataset,DataLoader
from tokenizers import Tokenizer
from tokenizers.model import WordLevel
from tokenizers.model import WordLevelTrainer
from tokenizers.pretokenizers import Whitespace
from dataset import BilingualDataset, casual_mask
from pathlib import Path
from models import build_model
from torch.utils.tensorboard import SummaryWriter
from config import get_config,get_weights_file_path

from tqdm import tqdm



#building the tokenizer
def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]
def build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='UNK'))
        tokenizer.pretokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],word_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang) ,trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

#loading data from hugging gace
def get_ds(config):
    ds_raw=load_dataset('cous books', f'{config['lang_src'].config['lang_tgt']}', split='train')

    #build the tokennixer
    tokenizer_src=build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt=build_tokenizer(config,ds_raw,config['lang_tgt'])
    #splitting the data
    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_size=random_split(ds_raw, [train_ds_size,val_ds_size])

    train_ds=BilingualDataset(train_ds,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BilingualDataset(val_ds,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src,max_len_tgt=0,0
    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids

        max_len_src=max(max_len_src, len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f" Maximum length of source sentence is {max_len_src}")
    print(f"Maximum length of target id is {max_len_tgt}")
    #creating dataloaders

    train_dataloader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=False)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt
        
       
def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_model(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model


def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,writer,num_examples=2):
    model.eval()
    source_texts=[]
    expected=[]
    predicted=[]

    console_width=50

    with torch.no_grad():
        for batch in validation_ds:
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)

            assert encoder_input.size(0)==1
    



def train_model(config):
    device= "cuda" if torch.cuda.is_available else "cpu"

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

    writer=SummaryWriter(config['experiment_name'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch,global_step=0,0

    if config['preload']:
        model_filename=get_weights_file_path(config,config['preload'])
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']

    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()

        batch_iterator=tqdm(train_dataloader,desc="processing epoch")

        for batch in batch_iterator:
            encoder_input=batch['encoder_input'].to(device)
            decoder_input=batch['decoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            decoder_mask=batch['decoder_mask'].to(device)

            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.encode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            proj_output=model.project(decoder_output)

            label=batch['label'].to(device)

            loss=loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({f"loss: f{loss.item():.3f}"})

            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        model_filename=get_weights_file_path(config, f"{epoch}")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        },model_filename)


if __name__ == '__main__':
    config=get_config()
    train_model()


