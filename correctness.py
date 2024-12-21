# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import os
import random
import logging

import torch

import transformers

from arguments import BenchmarkArguments, process_cli_arguments
from data import get_data
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import (
    SelfSpeculativeGenerationStrategy,
)

from self_speculation.dynamic_early_exit_generator import DynamicEarlyExitGenerationStrategy

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=48)
    )
    rank = int(os.environ["LOCAL_RANK"])
    
    random.seed(benchmark_arguments.seed)
    torch.manual_seed(benchmark_arguments.seed)

    if rank != 0:
        # only run on rank 0, we don't support parallel inference yet
        return

    local_model_path: str = benchmark_arguments.model

    # initialize model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        local_model_path, use_fast=False
    )
    config = transformers.LlamaConfig.from_pretrained(local_model_path)
    model = transformers.LlamaForCausalLM.from_pretrained(
        local_model_path,
        config=config,
        torch_dtype=torch.float16,
    )
    model.cuda()
    model.half()
    model.eval()

    # initialize generator
    spec_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=SelfSpeculativeGenerationStrategy(),
    )

    ar_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=AutoRegressiveGenerationStrategy(),
    )

    dee_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=DynamicEarlyExitGenerationStrategy(),
    )

    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        data_path=benchmark_arguments.data_path,
    )

    errors_spec = 0
    errors_dee = 0
    
    for i, example in enumerate(tqdm(evaluation_set)):
        spec_response: GenerationResult = spec_generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        ar_response: GenerationResult = ar_generator.generate(
            prompt=example.input,
            # generation config to use the full model
            generation_config=GenerationConfig(
                max_steps=generation_config.max_steps,
                exit_layer=-1,
                num_speculations=-1,
                generation_strategy="autoregressive",
            ),
        )
                # Get dynamic early exit response
        dee_response: GenerationResult = dee_generator.generate(
            prompt=example.input,
            generation_config=GenerationConfig(
                max_steps=generation_config.max_steps,
                confidence_threshold=generation_config.confidence_threshold,
                num_speculations=generation_config.num_speculations,
                generation_strategy="dynamic_early_exit",
            ),
        )

        if spec_response.decoded_prediction != ar_response.decoded_prediction:
            errors_spec += 1
            log.info("Error found")
            log.info(f"Spec response: {spec_response}")
            log.info(f"AR response: {ar_response}")
        
                # Compare dynamic early exit with autoregressive
        if dee_response.decoded_prediction != ar_response.decoded_prediction:
            errors_dee += 1
            log.info("Dynamic early exit error found")
            log.info(f"DEE response: {dee_response}")
            log.info(f"AR response: {ar_response}")

    metric_result = {
        "self_speculative": {
            "errors": errors_spec,
            "error_pct": errors_spec / len(evaluation_set)
        },
        "dynamic_early_exit": {
            "errors": errors_dee,
            "error_pct": errors_dee / len(evaluation_set)
        }
    }
    print(metric_result)

    with open(output_fname, "w") as f:
        json.dump(metric_result, f)


if __name__ == "__main__":
    args = process_cli_arguments()
    main(args.benchmark_arguments, args.generation_config, f"{args.benchmark_arguments.output_dir}/correctness_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
