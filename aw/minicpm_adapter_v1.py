"""Exact MiniCPM-o-4.5 vision-only path; no audio or thinking generation."""
import logging,math
import numpy as np
import torch
from PIL import Image,ImageOps
from .model import OmniAdapter
from .common import seed_all,paired_seed
from .sampling import deterministic_nucleus

class MiniCPMAdapter(OmniAdapter):
    def __init__(self,cfg):
        from transformers import AutoModel,AutoProcessor
        self.cfg=cfg;m=cfg['model']
        if m['id']!='openbmb/MiniCPM-o-4_5':raise ValueError('Wrong model')
        if not torch.cuda.is_available():raise RuntimeError('Allocated GPU required')
        self.dtype=getattr(torch,m['dtype'])
        self.model=AutoModel.from_pretrained(m['path'],trust_remote_code=True,local_files_only=True,
            torch_dtype=self.dtype,device_map={'':0},attn_implementation=m['attention'],
            init_vision=True,init_audio=False,init_tts=False).eval()
        self.processor=AutoProcessor.from_pretrained(m['path'],trust_remote_code=True,local_files_only=True)
        self.model.prepare_processor(processor=self.processor)
        self.layers=self.model.llm.model.layers
        assert len(self.layers)==m['layers'] and self.model.config.hidden_size==m['hidden_size']
        for p in self.model.parameters():p.requires_grad_(False)
        self.device=next(self.model.parameters()).device
        logging.info('MiniCPM native vision path, %s decoder layers; audio/TTS disabled per official documented mode',len(self.layers))

    def inputs(self,image_path,text):
        with Image.open(image_path) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        cap=self.cfg['model']['max_image_pixels']
        if image.width*image.height>cap:
            scale=math.sqrt(cap/(image.width*image.height))
            image=image.resize((max(1,int(image.width*scale)),max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
        messages=[{'role':'system','content':self.cfg['prompts']['system']},
                  {'role':'user','content':'<image>./</image>\n'+text}]
        prompt=self.processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
            use_tts_template=False,enable_thinking=False)
        batch=self.processor([prompt],[[image]],[[]],[[]],return_tensors='pt',max_length=8192,
            max_slice_nums=self.cfg['model']['max_slice_nums'],use_image_id=True,stream_input=False).to(self.device)
        batch.pop('image_sizes',None)
        if not len(batch['image_bound'][0]):raise RuntimeError('No image tokens')
        return batch

    @torch.inference_mode()
    def extract(self,image_path,text):
        inputs=self.inputs(image_path,text);mask=inputs['attention_mask'].bool();rows={}
        def make_hook(i):
            def hook(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                assert h.shape[1]==mask.shape[1],'Pooling mask alignment'
                rows[i]=h[0,mask[0]].float().mean(0).cpu().numpy()
            return hook
        handles=[layer.register_forward_hook(make_hook(i)) for i,layer in enumerate(self.layers)]
        try:
            embeddings,_=self.model.get_vllm_embedding(inputs)
            self.model.llm(inputs_embeds=embeddings,attention_mask=inputs['attention_mask'],use_cache=False,return_dict=True)
        finally:
            for handle in handles:handle.remove()
        return np.stack([rows[i] for i in range(len(self.layers))]),int(mask.sum())

    @torch.inference_mode()
    def generate(self,row,vectors=None,layers=(),emotion_prompt=None,anchor=None,budget='per_layer',seed=42,max_new_tokens=None):
        seed_all(paired_seed(row['image_id'],seed));inputs=self.inputs(row['image_path'],self.prompt(emotion_prompt,anchor))
        g=self.cfg['generation'];limit=max_new_tokens or g['max_new_tokens']
        with self.intervention(vectors,layers,self.cfg['steering']['alpha'],budget) as audit,deterministic_nucleus():
            _,outputs=self.model.generate(**inputs,tokenizer=self.processor.tokenizer,stream=False,
                max_new_tokens=limit,do_sample=g['do_sample'],temperature=g['temperature'],
                top_p=g['top_p'],top_k=g['top_k'],repetition_penalty=1.0,use_cache=True)
        # Official generate uses inputs_embeds: sequences contain NEW tokens only.
        tokens=outputs.sequences[0].cpu().tolist()
        assert len(tokens)<=limit
        return {'text':self.processor.tokenizer.decode(tokens,skip_special_tokens=True),
            'generated_token_ids':tokens,'generated_tokens':len(tokens),'input_tokens':inputs['input_ids'].shape[1],
            'injection_events':audit,'paired_seed':paired_seed(row['image_id'],seed)}
