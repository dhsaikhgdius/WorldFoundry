"""Module for base_models -> diffusion_model -> video -> step_video -> step_video_runtime -> stepvideo -> text_encoder -> clip.py functionality."""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os


class HunyuanClip(nn.Module):
    """
        Hunyuan clip code copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
        hunyuan's clip used BertModel and BertTokenizer, so we copy it.
    """
    def __init__(self, model_dir, max_length=77):
        """Init.

        Args:
            model_dir: The model dir.
            max_length: The max length.
        """
        super(HunyuanClip, self).__init__()
        
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        self.text_encoder = BertModel.from_pretrained(os.path.join(model_dir, 'clip_text_encoder'))
        
    @torch.no_grad
    def forward(self, prompts, with_mask=True):
        """Forward.

        Args:
            prompts: The prompts.
            with_mask: The with mask.
        """
        self.device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device),
            attention_mask=text_inputs.attention_mask.to(self.device) if with_mask else None,
        )
        return prompt_embeds.last_hidden_state, prompt_embeds.pooler_output
        
