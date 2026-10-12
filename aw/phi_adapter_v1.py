"""Phi replica: frozen public vision path, shared intervention/sampling protocol."""
import logging,math
import torch
from PIL import Image,ImageOps
from .model import OmniAdapter

class PhiAdapter(OmniAdapter):
    def __init__(self,cfg):
        from transformers import AutoModelForCausalLM,AutoProcessor
        self.cfg=cfg;m=cfg['model']
        if m['id']!='microsoft/Phi-4-multimodal-instruct':raise ValueError('Wrong replica model')
        if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
        self.dtype=getattr(torch,m['dtype'])
        self.processor=AutoProcessor.from_pretrained(m['path'],trust_remote_code=True,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(m['path'],torch_dtype=self.dtype,device_map={'':0},
            trust_remote_code=True,local_files_only=True,_attn_implementation=m['attention']).eval()
        self.layers=self.model.model.layers
        assert len(self.layers)==m['layers'] and self.model.config.hidden_size==m['hidden_size']
        for p in self.model.parameters():p.requires_grad_(False)
        self.device=next(self.model.parameters()).device
        logging.info('Phi frozen vision model loaded: %s decoder layers',len(self.layers))

    def inputs(self,image_path,text):
        with Image.open(image_path) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        # Same configured input pixel budget; native image tokenization differs by model.
        cap=self.cfg['model']['max_image_pixels']
        if image.width*image.height>cap:
            scale=math.sqrt(cap/(image.width*image.height))
            image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
        prompt='<|system|>'+self.cfg['prompts']['system']+'<|end|><|user|><|image_1|>'+text+'<|end|><|assistant|>'
        batch=self.processor(text=prompt,images=image,return_tensors='pt')
        # Official InputMode.VISION = 1; ensure no silent text-only fallback.
        if 'input_image_embeds' not in batch:raise RuntimeError('Missing image embedding inputs')
        mode=int(batch['input_mode'].reshape(-1)[0])
        if mode!=1:raise RuntimeError(f'Unexpected Phi input mode {mode}')
        return {k:(v.to(device=self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device))
                if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
