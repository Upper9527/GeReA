import pdb
from torch.utils.checkpoint import checkpoint
import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from .modeling_t5 import VLT5, VisualEmbedding 
import numpy as np
from .vl_encoder import *

class FiDT5(VLT5):
    def __init__(self, config, opt=None):
        super().__init__(config)
        self.wrap_encoder(config=config, opt=opt)
        self.opt = opt

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, vis_inputs=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            vis_inputs=vis_inputs,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, vis_inputs, attention_mask, max_length, num_beams=None, num_return_sequences=None, early_stopping=None):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            vis_inputs=vis_inputs,
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=early_stopping
        )

    def wrap_encoder(self, config, opt):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, config, opt)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(config=self.config, opt=self.opt)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, config, opt, n_block=9, use_checkpoint=False):
        super().__init__()
        self.config = config
        self.encoder = encoder
        if opt:
            use_checkpoint = opt.use_checkpoint
            n_block = opt.n_block
        self.visual_embedding = VisualEmbedding(config)
        self.vis_encoder = build_vl_transformer(1024, 0.1, 8, 1024, n_block)
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, vis_inputs=None, **kwargs,):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, None, **kwargs)
        
        vis_feat = vis_inputs
        # boxes = vis_inputs[1]
        visual_emb = self.visual_embedding(vis_feat,  None, None)
        visual_emb = self.vis_encoder(visual_emb)

        outputs0 = torch.cat((visual_emb, outputs[0].view(bsz, self.n_passages*passage_length, -1)), 1)
        outputs = (outputs0,) + outputs[1:]
        return outputs

    def set_grad(self):
        for name, param in self.vis_encoder.named_parameters():
            if param.grad == None:
                pdb.set_trace()
                param.requires_grad = True
                param.grad = 0

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

