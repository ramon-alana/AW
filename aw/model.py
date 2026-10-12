"""Primary Qwen Omni image/text adapter. No alternate-model fallback."""
from contextlib import contextmanager
import logging
import numpy as np
import torch
from PIL import Image,ImageOps
from .sampling import deterministic_nucleus
from .common import seed_all,paired_seed

class OmniAdapter:
    def __init__(self,cfg):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
        self.cfg=cfg; m=cfg['model']
        if not torch.cuda.is_available():raise RuntimeError('Actual model work requires an allocated GPU')
        dtype=getattr(torch,m['dtype'])
        self.processor=Qwen2_5OmniProcessor.from_pretrained(m['path'],local_files_only=True)
        self.processor.image_processor.max_pixels=m['max_image_pixels']
        self.model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(m['path'],torch_dtype=dtype,device_map={'':0},attn_implementation=m['attention'],local_files_only=True).eval()
        self.layers=self.model.model.layers
        if len(self.layers)!=m['layers']:raise RuntimeError('Unexpected decoder count')
        for param in self.model.parameters():param.requires_grad_(False)
        self.device=next(self.model.parameters()).device
        logging.info('Loaded actual Qwen Omni Thinker with %s decoder layers',len(self.layers))

    def inputs(self,image_path,text):
        image=ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
        messages=[{'role':'system','content':[{'type':'text','text':self.cfg['prompts']['system']}]},{'role':'user','content':[{'type':'image','image':str(image_path)},{'type':'text','text':text}]}]
        rendered=self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        # Omni 4.52 returns a one-element list even for one conversation.
        texts=[rendered] if isinstance(rendered,str) else rendered
        if not isinstance(texts,list) or len(texts)!=1 or not isinstance(texts[0],str):
            raise TypeError('Expected exactly one rendered text conversation')
        batch=self.processor(text=texts,images=[image],return_tensors='pt',padding=True)
        return {k:v.to(self.device) if hasattr(v,'to') else v for k,v in batch.items()}

    def prompt(self,emotion=None,anchor=None):
        text=self.cfg['prompts']['base']
        if anchor:text+='\n'+self.cfg['prompts']['anchor_prefix'].format(anchor=anchor)
        if emotion:text+='\n'+self.cfg['prompts']['condition'].format(emotion=emotion)
        return text

    @torch.inference_mode()
    def extract(self,image_path,text):
        inputs=self.inputs(image_path,text);mask=inputs['attention_mask'].bool();rows={}
        def make_hook(i):
            def hook(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                rows[i]=h[0,mask[0]].float().mean(0).cpu().numpy()
            return hook
        handles=[layer.register_forward_hook(make_hook(i)) for i,layer in enumerate(self.layers)]
        try:self.model(**inputs,use_cache=False,return_dict=True)
        finally:
            for h in handles:h.remove()
        return np.stack([rows[i] for i in range(len(self.layers))]),int(mask.sum())

    @contextmanager
    def intervention(self,vectors,layers,alpha,budget='per_layer'):
        if len(set(layers))!=len(layers) or any(l<1 or l>len(self.layers) for l in layers):raise ValueError('Bad layer set')
        if budget not in ('per_layer','sum_relative_l2'):raise ValueError('Unknown budget')
        handles=[];receipt=[];scale=alpha/(len(layers) if budget=='sum_relative_l2' and layers else 1)
        def make_hook(index):
            v=torch.as_tensor(vectors[index-1],device=self.device,dtype=torch.float32)
            if not torch.isfinite(v).all() or not torch.isclose(v.norm(),torch.tensor(1.,device=self.device),atol=1e-4):raise ValueError('Non-unit direction')
            def hook(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                edited=h.clone();last=h[:,-1,:].float();delta=scale*last.norm(dim=-1,keepdim=True)*v
                edited[:,-1,:]=(last+delta).to(h.dtype)
                receipt.append({'layer':index,'sequence_length':h.shape[1],'relative_norm':float(delta.norm()/(last.norm()+1e-12))})
                return (edited,*output[1:]) if isinstance(output,tuple) else edited
            return hook
        try:
            for index in layers:handles.append(self.layers[index-1].register_forward_hook(make_hook(index)))
            yield receipt
        finally:
            for h in handles:h.remove()

    @torch.inference_mode()
    def generate(self,row,vectors=None,layers=(),emotion_prompt=None,anchor=None,budget='per_layer',seed=42,max_new_tokens=None):
        seed_all(paired_seed(row['image_id'],seed));inputs=self.inputs(row['image_path'],self.prompt(emotion_prompt,anchor))
        g=self.cfg['generation'];limit=max_new_tokens or g['max_new_tokens']
        with self.intervention(vectors,layers,self.cfg['steering']['alpha'],budget) as audit,deterministic_nucleus():
            ids=self.model.generate(**inputs,max_new_tokens=limit,do_sample=g['do_sample'],temperature=g['temperature'],top_p=g['top_p'],top_k=g['top_k'],use_cache=True)
        tokens=ids[0,inputs['input_ids'].shape[1]:].cpu().tolist()
        return {'text':self.processor.tokenizer.decode(tokens,skip_special_tokens=True),'generated_token_ids':tokens,'generated_tokens':len(tokens),'input_tokens':inputs['input_ids'].shape[1],'injection_events':audit,'paired_seed':paired_seed(row['image_id'],seed)}
