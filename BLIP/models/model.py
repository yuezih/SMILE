import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertLMHeadModel
from transformers import BertTokenizer


import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

import pdb

class CapModel(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):

        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.vocab_emb = None

    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id

        # # First-token Shifting: Change the first token 'word' to '##word'
        # for i in range(text.input_ids.size(0)):
        #     text.input_ids[i, self.prompt_length] = self.tokenizer.convert_tokens_to_ids('##' + self.tokenizer.convert_ids_to_tokens(text.input_ids[i,self.prompt_length].item()))

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )
        
        # # mle
        # mle_loss = decoder_output.loss
        
        label = text.input_ids[:, self.prompt_length:].contiguous()
        bs = text.input_ids.size(0)
        N = label.size(1)
        vs = self.text_decoder.config.vocab_size
        logits = decoder_output.logits[:, self.prompt_length-1:-1]
        
        # smile
        mask = torch.zeros(bs, vs).to(logits.device).scatter_(1, label, True)
        mask[:, 0] = 0
        mask = mask.unsqueeze(1).expand(-1, N, -1).clone()
        mask[:, 0, :] = 1 # mle on first token
        selected_logits = logits.masked_fill(mask == 0, -1e9)
        smile_loss = F.cross_entropy(selected_logits.view(-1, vs), label.view(-1), ignore_index=0, reduction='mean')

        # # reverse smile
        # reverse_mask = torch.ones(bs, vs).to(logits.device).scatter_(1, label, False)
        # reverse_mask = reverse_mask.unsqueeze(1).expand(-1, N, -1).clone()
        # reverse_mask.scatter_(2, label.unsqueeze(-1), 1)
        # reverse_mask[:, 0, :] = 1 # mle on first token
        # reverse_selected_logits = logits.masked_fill(reverse_mask == 0, -1e9)
        # reverse_smile_loss = F.cross_entropy(reverse_selected_logits.view(-1, vs), label.view(-1), ignore_index=0, reduction='mean')

        # # random sample (efficient implementation)
        # sample_num = 10
        # rand_indices = torch.randint(vs, (bs, N, sample_num)).to(label.device)
        # rand_indices_with_label = torch.cat((rand_indices, label.unsqueeze(2)), dim=2) # (bs, N, sample_num + 1)
        # batch_indices = torch.arange(bs)[:, None, None].expand(bs, N, sample_num + 1)
        # seq_indices = torch.arange(N)[None, :, None].expand(bs, N, sample_num + 1)
        # random_mask = torch.zeros(bs, N, vs).to(label.device)
        # random_mask[batch_indices, seq_indices, rand_indices_with_label] = 1
        # random_mask[:, :, 0] = 0
        # random_selected_logits = logits.masked_fill(mask == 0, -1e9)
        # random_smile_loss = F.cross_entropy(random_selected_logits.view(-1, vs), label.view(-1), ignore_index=0, reduction='mean')

        loss = smile_loss
        # loss = 0.5 * reverse_smile_loss + 0.5 * mle_loss

        return loss
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
            # caption = self.tokenizer.decode(output[4:], skip_special_tokens=True)
            # captions.append(caption)
        return captions
    

def caption_model(pretrained='',**kwargs):
    model = CapModel(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
    return model

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    

    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg