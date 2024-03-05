import torch

@torch.no_grad()
def greedy_decoding(sent, model, tokenizer_src, tokenizer_tgt,max_seq_length):
    model.eval()
    device = model.device
    src = tokenizer_src(sent, return_tensors="pt")['input_ids'].to(device)
    result = torch.ones(1, 1).fill_(tokenizer_tgt.cls_token_id).type(torch.long).to(device)

    src_mask, _ = model.generate_mask(src, result)
    src_embed, _ = model.create_embeddings(src, result)
    memory, enc_attention_map = model.encoder(src_embed,src_mask)
    for i in range(max_seq_length-1):
        _, tgt_mask = model.generate_mask(src,result)
        _, tgt_embed = model.create_embeddings(src, result)
        decoder_outs ,decoder_attention_map, cross_attention_map =  model.decoder(memory,tgt_embed,tgt_mask)
        prob = model.generator(decoder_outs)
        next_word = prob.argmax(dim=-1)[0][-1].item()
        result = torch.cat([result, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        if next_word == tokenizer_tgt.sep_token_id :  # Assuming 102 corresponds to [SEP]
            break
        generated_sentence = tokenizer_tgt.decode(result.detach().cpu()[0],skip_special_tokens=True)
    return {
        'generated_sentence':generated_sentence,
        'tokens':result,
        'enc_att_map':enc_attention_map,
        'dec_att_map':decoder_attention_map,
        'cross_att_map':cross_attention_map
    }