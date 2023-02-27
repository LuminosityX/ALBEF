import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # config['image_res'] = 256
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),             # interpolation 插值法
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':                                                      # True
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
    
    '''
    def pre_caption(caption,max_words):
        caption = re.sub(                                                       # re.sub  正则表达式的替换
            r"([,.'!?\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption
        
    class re_train_dataset(Dataset):
        def __init__(self, ann_file, transform, image_root, max_words=30):        
            self.ann = []
            for f in ann_file:                                                  # train_file 是一个list
                self.ann += json.load(open(f,'r'))
            self.transform = transform
            self.image_root = image_root
            self.max_words = max_words
            self.img_ids = {}   

            n = 0
            for ann in self.ann:
                img_id = ann['image_id']
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1    

        def __len__(self):
            return len(self.ann)

        def __getitem__(self, index):    

            ann = self.ann[index]

            image_path = os.path.join(self.image_root,ann['image'])        
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)

            caption = pre_caption(ann['caption'], self.max_words)           # max_words = 30

            return image, caption, self.img_ids[ann['image_id']]
    
    

    class re_eval_dataset(Dataset):
        def __init__(self, ann_file, transform, image_root, max_words=30):        
            self.ann = json.load(open(ann_file,'r'))
            self.transform = transform
            self.image_root = image_root
            self.max_words = max_words 

            self.text = []
            self.image = []
            self.txt2img = {}
            self.img2txt = {}

            txt_id = 0
            for img_id, ann in enumerate(self.ann):
                self.image.append(ann['image'])
                self.img2txt[img_id] = []
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption,self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1

        def __len__(self):
            return len(self.image)

        def __getitem__(self, index):    

            image_path = os.path.join(self.image_root, self.ann[index]['image'])        
            image = Image.open(image_path).convert('RGB')    
            image = self.transform(image)  

            return image, index                                          # 只返回image ???
    '''
    
    # 在Flickr的json中，train与eval的不同
    elif dataset=='re':   
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):                                  # datasets 本身就是list, [True], num_tasks, global_rank
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):                                               # 对于Retrieval也是就1个，但是是因为eval不需要sampler
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):          
    '''[train_dataset, val_dataset, test_dataset],samplers,
       batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,  # 32, 64, 54
       num_workers=[4,4,4],
       is_trains=[True, False, False], 
       collate_fns=[None,None,None]'''
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
