# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import transformers

@dataclass
class ForwardResult:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    exit_query_cache: Optional[List[torch.Tensor]] = None

# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = transformers.generation.logits_process.TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = transformers.generation.logits_process.TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits

def decode_next_token(
    logits: torch.Tensor,
    token_idx: int = None,
    sample: Optional[bool] = False,
    temperature: Optional[float] = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> torch.Tensor:
    if token_idx:
        logits = logits[:, -1, :]

    if not sample:
        next_token = logits.argmax(dim=-1)
        return next_token, None
    else:
        if not token_idx:
            logits.squeeze_(dim=0)
        filtered_logits = top_k_top_p_filtering(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        if not token_idx:
            next_token.transpose_(1, 0)
        return next_token, probabilities


def crop_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    maximum_length: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(past_key_values)):
        if past_key_values[idx] is None or past_key_values[idx][0] == [] or past_key_values[idx][0] is None:
            break
        new_past.append(
            (
                past_key_values[idx][0][:, :, :maximum_length, :],
                past_key_values[idx][1][:, :, :maximum_length, :],
            )
        )
    past_key_values = tuple(new_past)
    return past_key_values


# Our forward_early(...) and forward_remainder(...) functions currently use transformers library's legacy KV cache implementation that is less efficient.
# To ensure an apples to apples comparison, we created this forward function to use in autoregressive decoding to ensure it uses the same KV cache implementation instead.
# FIXME: update forward_early(...) and forward_remainder(...) to use the updated more efficient KV cache implementation.
def forward(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    for decoder_layer in model.model.layers:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values
    )


# TODO: update forward_early(...) to use transformers' new KV cache implementation rather than legacy.
def forward_early(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    inputs_embeds = model.model.embed_tokens(input_ids)
    attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds
    for decoder_layer in model.model.layers[:exit_layer]:
        hidden_states, past_key_values = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            padding_mask=None,
        )

    past_key_values = past_key_values.to_legacy_cache()

    # next_cache = next_decoder_cache
    if exit_query_cache is None:
        exit_query_cache = hidden_states
    else:
        exit_query_cache = torch.cat([exit_query_cache, hidden_states], dim=1)

    hidden_states = model.model.norm(hidden_states)

    logits = model.lm_head(hidden_states)
    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache
    )


# TODO: update forward_remainder(...) to use transformers' new KV cache implementation rather than legacy.
def forward_remainder(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    exit_layer: int,
    exit_query_cache: Optional[List[torch.Tensor]],
) -> ForwardResult:
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    num_tokens_to_generate: int = 1
    seq_length_with_past = seq_length
    draft_past_key_values_length: int = 0
    full_past_key_values_length: int = 0

    if past_key_values is not None and past_key_values[0] is not None:
        # it's okay to use the first layer because the draft model necessairly computes it
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        # the total sequence length is the past key values since that includes the draft tokens

        # the last layer should not have been skipped, we can get this to check how many of the tokens have gone through full
        # verification
        if len(past_key_values) == len(model.model.layers):
            full_past_key_values_length = past_key_values[-1][0].shape[2]
        else:
            # we have not done a full pass yet so the history is 0
            full_past_key_values_length = 0

        seq_length_with_past = num_tokens_to_generate + draft_past_key_values_length
    past_key_values = transformers.cache_utils.DynamicCache.from_legacy_cache(past_key_values)

    inputs_embeds = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(
        full_past_key_values_length,
        seq_length_with_past,
        dtype=torch.long,
        device=device,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    attention_mask = input_ids.new_ones(
        (batch_size, seq_length_with_past),
        dtype=torch.bool,
    )
    early_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, num_tokens_to_generate),
        inputs_embeds,
        draft_past_key_values_length,
    )

    full_attention_mask = _prepare_decoder_attention_mask(
        model,
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        full_past_key_values_length,  # we have no past for the full model
    )

    next_decoder_cache = []
    hidden_states = inputs_embeds
    # TODO simplify
    full_hidden_states: Optional[torch.FloatTensor] = None
    for idx, decoder_layer in enumerate(model.model.layers):
        is_early_exit = idx < exit_layer
        past_key_value = (
            past_key_values[idx]
            if (past_key_values is not None and idx < len(past_key_values))
            else None
        )
        if is_early_exit:
            # early hidden states: B x num_gen x C
            early_hidden_states = hidden_states[:, -num_tokens_to_generate:]
            early_position_ids = position_ids[:, -num_tokens_to_generate:]
            hidden_states, past_key_values = decoder_layer(
                early_hidden_states,
                attention_mask=early_attention_mask,
                position_ids=early_position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )
        else:
            if full_hidden_states is None and exit_query_cache is not None:
                # first time seeing the full hidden states, we need to rely on the
                # query cache
                # only use if exit query cache exists, if not this is our first call
                full_hidden_states = torch.cat(
                    [exit_query_cache, hidden_states[:, -num_tokens_to_generate:]],
                    dim=1,
                )
            else:
                # we already have seen the fully hidden states we can re-use them now
                full_hidden_states = hidden_states
            hidden_states, past_key_values = decoder_layer(
                full_hidden_states,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                padding_mask=None,
            )

    past_key_values = past_key_values.to_legacy_cache()
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return ForwardResult(
        logits=logits, past_key_values=past_key_values, exit_query_cache=exit_query_cache
    )

def determine_exit_layer(
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        confidence_threshold: float,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> int:
    """Determine exit layer for a batch of tokens based on KL divergence."""
    draft_input_ids = input_ids.clone()
    device = draft_input_ids.device
    draft_batch_size, draft_seq_length = draft_input_ids.shape

    draft_seq_length_with_past = draft_seq_length
    draft_past_key_values_length = 0

    if past_key_values is not None:
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        draft_seq_length_with_past = draft_seq_length_with_past + draft_past_key_values_length

    # Set up position_ids
    draft_position_ids = torch.arange(
        draft_past_key_values_length,
        draft_seq_length + draft_past_key_values_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).view(-1, draft_seq_length)

    # Set up attention mask
    draft_attention_mask = draft_input_ids.new_ones((draft_batch_size, draft_seq_length_with_past))
    draft_inputs_embeds = model.model.embed_tokens(draft_input_ids)
    draft_attention_mask = _prepare_decoder_attention_mask(
        model,
        draft_attention_mask,
        (draft_batch_size, draft_seq_length),
        draft_inputs_embeds,
        draft_past_key_values_length,
    )

    draft_hidden_states = draft_inputs_embeds
    draft_last_logits = None
    exit_layer = len(model.model.layers)-1  # Default to using all layers

    for idx, decoder_layer in enumerate(model.model.layers):
        draft_hidden_states = decoder_layer(
            draft_hidden_states,
            attention_mask=draft_attention_mask,
            position_ids=draft_position_ids,
            past_key_value=None,  # Don't need cache for probing
            output_attentions=False,
            use_cache=False,
            padding_mask=None,
        )[0]

        if idx >= 7:
            draft_current_hidden = model.model.norm(draft_hidden_states)
            draft_current_logits = model.lm_head(draft_current_hidden)

            if draft_last_logits is not None:
                current_probs = torch.softmax(draft_current_logits[:, -1], dim=-1)
                last_probs = torch.softmax(draft_last_logits[:, -1], dim=-1)

                # Only consider top K tokens for stability measurement
                k = 10  # We can adjust this
                current_top = torch.topk(current_probs, k)
                last_top = torch.topk(last_probs, k)

                # Calculate difference only on top tokens
                top_indices = torch.unique(torch.cat([current_top.indices, last_top.indices]))
                prob_diff = torch.abs(current_probs[..., top_indices] - last_probs[..., top_indices]).mean()

                confidence = torch.max(current_probs, dim=-1)[0]

                #print(f"\nLayer {idx + 1}:")
                #print(f"Prob diff (top tokens) = {prob_diff:.6f}, Confidence = {confidence.mean():.4f}")

                # Adjust threshold since we're looking at fewer tokens
                if prob_diff < 0.01 and confidence.mean() > confidence_threshold:
                    #print(f"Exit criteria met at layer {idx + 1}")
                    return idx + 1

            draft_last_logits = draft_current_logits.detach().clone()
    return exit_layer


def determine_shared_exit_layer(
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        confidence_threshold: float,
        num_speculations: int
) -> int:
    """Determine a shared exit layer for a batch of tokens."""
    # First generate all tokens
    with torch.no_grad():
        draft_input_ids = input_ids.clone()
        all_tokens = [draft_input_ids]

        for _ in range(num_speculations - 1):
            logits = model(draft_input_ids, use_cache=False).logits
            next_token = torch.argmax(logits[:, -1], dim=-1)
            draft_input_ids = torch.cat([draft_input_ids, next_token.unsqueeze(0)], dim=-1)
            all_tokens.append(next_token.unsqueeze(0))

        # Concatenate all tokens
        full_sequence = torch.cat(all_tokens, dim=-1)

        # Determine exit layer for full sequence
        exit_layer = determine_exit_layer(model, full_sequence, confidence_threshold)
        if exit_layer is not None and 1 <= exit_layer < len(model.model.layers):
            return exit_layer
        return len(model.model.layers)-1
    
def determine_entropy_shared_exit_layer(
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        entropy_threshold: float,
        num_speculations: int
) -> int:
    """Determine a shared exit layer for a batch of tokens."""
    # First generate all tokens
    with torch.no_grad():
        draft_input_ids = input_ids.clone()
        all_tokens = [draft_input_ids]

        for _ in range(num_speculations - 1):
            logits = model(draft_input_ids, use_cache=False).logits
            next_token = torch.argmax(logits[:, -1], dim=-1)
            draft_input_ids = torch.cat([draft_input_ids, next_token.unsqueeze(0)], dim=-1)
            all_tokens.append(next_token.unsqueeze(0))

        # Concatenate all tokens
        full_sequence = torch.cat(all_tokens, dim=-1)

        # Determine exit layer for full sequence
        exit_layer = determine_entropy_exit_layer(model, full_sequence, entropy_threshold, num_speculations)
        #(f"Exit layer: {exit_layer}")
        if exit_layer is not None and 1 <= exit_layer < len(model.model.layers):
            return exit_layer
        
        #print(f"Exit layer not found, using all layers: {len(model.model.layers)-1}")
        return len(model.model.layers)-1

def determine_entropy_exit_layer(
    model: transformers.LlamaForCausalLM,
    input_ids: torch.Tensor,
    entropy_threshold: float,
    num_speculations: int,
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> int:
    """
    Determines the shared exit layer for a batch of tokens based on entropy thresholds.

    Args:
        model (transformers.LlamaForCausalLM): The model used for token generation.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_length].
        entropy_threshold (float): The entropy threshold for determining early exit.
        num_speculations (int): Number of speculative tokens to generate.
        past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]): Past key values for efficient computation.

    Returns:
        int: The shared exit layer index, or the last layer if no early exit is determined.
    """
    draft_input_ids = input_ids.clone()
    device = draft_input_ids.device
    draft_batch_size, draft_seq_length = draft_input_ids.shape

    draft_seq_length_with_past = draft_seq_length
    draft_past_key_values_length = 0

    if past_key_values is not None:
        draft_past_key_values_length = past_key_values[0][0].shape[2]
        draft_seq_length_with_past = draft_seq_length_with_past + draft_past_key_values_length

    # Set up position_ids
    draft_position_ids = torch.arange(
        draft_past_key_values_length,
        draft_seq_length + draft_past_key_values_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).view(-1, draft_seq_length)

    # Set up attention mask
    draft_attention_mask = draft_input_ids.new_ones((draft_batch_size, draft_seq_length_with_past))
    draft_inputs_embeds = model.model.embed_tokens(draft_input_ids)
    draft_attention_mask = _prepare_decoder_attention_mask(
        model,
        draft_attention_mask,
        (draft_batch_size, draft_seq_length),
        draft_inputs_embeds,
        draft_past_key_values_length,
    )

    draft_hidden_states = draft_inputs_embeds
    draft_last_logits = None
    exit_layer = len(model.model.layers)-1  # Default to using all layers

    for idx, decoder_layer in enumerate(model.model.layers):
        draft_hidden_states = decoder_layer(
            draft_hidden_states,
            attention_mask=draft_attention_mask,
            position_ids=draft_position_ids,
            past_key_value=None,  # Don't need cache for probing
            output_attentions=False,
            use_cache=False,
            padding_mask=None,
        )[0]

        if idx >= 7:
            draft_current_hidden = model.model.norm(draft_hidden_states)
            draft_current_logits = model.lm_head(draft_current_hidden)

            if draft_last_logits is not None:
                current_probs = torch.softmax(draft_current_logits[:, -1], dim=-1)
                current_probs = current_probs.to(torch.float32)  # Switch to float32 for stability
                current_probs = current_probs + 1e-6            # Add small epsilon to avoid zeros
                current_probs = current_probs / current_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
                entropy = -torch.sum(current_probs * torch.log(current_probs), dim=-1).mean().item()                
                #print(f"\nLayer {idx + 1}:")
                #print(f"\ncurrent_probs {current_probs}:")
                #print(f"entropy (top tokens) = {entropy:.6f}")

                # Adjust threshold since we're looking at fewer tokens
                if entropy < entropy_threshold:
                    #print(f"Exit criteria met at layer {idx + 1}")
                    return idx + 1

            draft_last_logits = draft_current_logits.detach().clone()
    return exit_layer