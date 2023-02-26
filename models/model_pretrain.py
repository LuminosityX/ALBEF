'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial                                         # 把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,                                                # pretrain: config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']              # 0.15
        embed_dim = config['embed_dim']                               # 256
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:                                                 # True
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)                  # check_hash : True,则文件名要加上hash值
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)        # 这里ViT加载的是DeiT的权重
            print(msg)          
            
        vision_width = config['vision_width']                         # 768
        '''
        {
          "architectures": [
            "BertForMaskedLM"
          ],
          "attention_probs_dropout_prob": 0.1,
          "hidden_act": "gelu",
          "hidden_dropout_prob": 0.1,
          "hidden_size": 768,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "layer_norm_eps": 1e-12,
          "max_position_embeddings": 512,
          "model_type": "bert",
          "num_attention_heads": 12,
          "num_hidden_layers": 12,
          "pad_token_id": 0,
          "type_vocab_size": 2,
          "vocab_size": 30522,
          "fusion_layer": 6,
          "encoder_width": 768
        }
        '''
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size             # 768
        self.vision_proj = nn.Linear(vision_width, embed_dim)         # 768 -> 256
        self.text_proj = nn.Linear(text_width, embed_dim)             # 768 -> 256

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])    # 0.07
        self.queue_size = config['queue_size']                       # 65536
        self.momentum = config['momentum']                           # 0.995
        self.itm_head = nn.Linear(text_width, 2)                     # 768 -> 2

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        self.text_proj_m = nn.Linear(text_width, embed_dim)                                 # 与前面的都一致
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()                                                                  # 将原模型的权重复制到动量模型，并且将动量模型requires_grad = False

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))        # register_buffer 希望state_dicts保存该参数，但是该参数又不会更新
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))         # [256, 65536]
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)



    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)                                                       # 依旧0.07
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)   # [batch_size, patch_number]

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)                # [batch_size, 256]

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,           # *** 这里应该得到的是前6层的输出   ***           
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                                   # [b, 256]
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()                                                                          # 更新动量模型
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)           # [256, 65536+b]                                 
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)              # [256, 65536+b]  

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp                                             # @即矩阵相乘 [b, 65536+b]
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)                                                                    # 对角线填充

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
        
        # sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        # F.log_softmax(sim_i2t, dim=1)   *  alpha * F.softmax(sim_i2t_m, dim=1)    为动量蒸馏 损失，文中说的是KL散度，但这里应该算的是交叉熵
        # F.log_softmax(sim_i2t, dim=1)   *  (1 - alpha) * sim_targets  这里算的应该就是ITC的损失函数，就是交叉熵损失函数
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean() 
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,                                   # [b, l, dim]
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,                               # [b, p, dim]
                                        encoder_attention_mask = image_atts,                                # 全1 [b, p]
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)                                                   # [b ,b]
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)                                                                   # 对角线全是0
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()                                          # 从tensor的每一行中采样，第一个参数就是采样的概率，第二个参数就是采样的个数，返回indice
            image_embeds_neg.append(image_embeds[neg_idx])                                                 # 直接取出每一个text的negative image features
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)                                                  # 跟上面的image一样

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)                                 # [2b, l, d]
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     
        # *** 注意，这里图像文本，所对应的cat是反的，text_embeds对image_embeds_neg， text_embeds_neg对的是image_embeds
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)                               # [2b, p, d ]  
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all, 
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )                         

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0) # [3b, 768]
        vl_output = self.itm_head(vl_embeddings)                                                                    # [3b, 2]

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                                          # [b, l]
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,  # vocab_size=30522, 
                                      probability_matrix = probability_matrix)                                       # [b, l] = 0.15
        
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm, loss_ita, loss_itm  

        

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)          # 0.995
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):                       # [b, 256]
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)                              # 应该是将分布式训练中产生的所有tensor都同时放入队列
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)                                                # 最开始为0 
        assert self.queue_size % batch_size == 0  # for simplicity               # 65536 // 64 = 1024

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr                                                  # 更新队列现在存到哪了
        
     # input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels, probability_matrix = probability_matrix # vocab_size=30522, [b, l] = 0.15
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                              # True   
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:                                                 # True
            targets[~masked_indices] = -100 # We only compute loss on masked tokens      # 后面在计算loss时，xbert代码里会使用到这的-100        

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)                            
    return output

