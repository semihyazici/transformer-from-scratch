from modules.Transformer import get_model
from modules.config import TransformerConfig
from tqdm import trange
import torch
import torch.nn as nn
from transformers import AutoTokenizer




def train_loop(model, loss_fn, optim, dataloader,tokenizer_src,tokenizer_tgt,save_path,n_steps = 100000):
    device = model.device
    cum_loss = 0.0
    progress_bar = trange(n_steps, desc="Training",leave=True)
        
    for step in progress_bar:


        batch_tgt, batch_src = next(iter(dataloader))

        src= tokenizer_src(batch_src,return_tensors="pt",padding=True )['input_ids'].to(device)
        tgt= tokenizer_tgt(batch_tgt,return_tensors="pt",padding=True )['input_ids'].to(device)
        

        tgt_input = tgt[:,:-1]
        optim.zero_grad()
        result = model(src,tgt_input,) 

        preds = result['logits']

        tgt_out = tgt[:,1:]
        
        loss = loss_fn(preds.reshape(-1, preds.shape[-1]), tgt_out.reshape(-1))
        cum_loss += loss.item()
        loss.backward()
        optim.step()

        progress_bar.set_description(f"Loss={(cum_loss / (step+1)):.4f}")

        progress_bar.update(1)

        
    torch.save(model.state_dict(),save_path)


def train(tokenizer_src_path,tokenizer_tgt_path,dataloader,save_path,n_steps=10000, device='cuda',model_from_pretrained = None ):

    tokenizer_src = AutoTokenizer.from_pretrained(tokenizer_src_path)
    tokenizer_tgt = AutoTokenizer.from_pretrained(tokenizer_tgt_path)


    config = TransformerConfig()
    model = get_model(config=config, vocab_size_src=tokenizer_src.vocab_size,
                       vocab_size_tgt=tokenizer_tgt.vocab_size,device=device)
    
    if model_from_pretrained is not None:
        model.load_state_dict(torch.load(model_from_pretrained))
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)
    optim = torch.optim.Adam(params=model.parameters(),lr=0.0001, betas=(0.9, 0.98),eps=10e-9)
    train_loop(model=model, loss_fn=loss_fn, optim=optim, dataloader=dataloader,
               tokenizer_src=tokenizer_src,tokenizer_tgt=tokenizer_tgt,n_steps=n_steps,save_path=save_path)
    


