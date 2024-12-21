# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import transformers
from datetime import datetime
import os
import tabulate
import torch

from arguments import Arguments, simple_parse_args_string
from benchmark import benchmark, load_model_and_tokenizer, process_cli_arguments, setup, BenchmarkArguments
from self_speculation.generator_base import (
    GenerationConfig,
)
from self_speculation.entropy_based_generator import EntropyExitGenerationStrategy
def sweep(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    results: List[Dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    # TODO: make start, end, step arguments to the script
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    if generation_config.generation_strategy == "self_speculative":
        for exit_layer in range(1, len(model.model.layers) // 2, 1):
            for num_speculations in range(1, 13, 1):
                generation_config.exit_layer = exit_layer
                generation_config.num_speculations = num_speculations

                metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

                results.append({
                    "exit_layer": exit_layer,
                    "num_speculations": num_speculations,
                    "acceptance_rate": metric_result['acceptance_rate']['mean'],
                    "total_time": metric_result['total_time']['mean'],
                    "time_per_token": metric_result['time_per_token']['mean'],
                    "tokens_per_second": metric_result['tokens_per_second']['mean'],
                })
                df = pd.DataFrame(results) 
                # Update table every iteration
                df.to_csv(output_fname, index=False)
                print(f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}")

    elif generation_config.generation_strategy == "dynamic_early_exit":
        # Sweep confidence threshold and num_speculations
        for confidence_threshold in np.arange(0.01, 0.5, 0.02):  # 0.01 to 0.5
            for num_speculations in range(1, 7, 2):
                generation_config.confidence_threshold = confidence_threshold
                generation_config.num_speculations = num_speculations

                metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

                # Get average exit layer from results
                #avg_exit_layer = sum(metric_result.get('exit_layers', [])) / len(metric_result.get('exit_layers', [1]))
                # Debug print entire metric_result
                print("\nFull metric_result:")
                print(metric_result)

                results.append({
                    "strategy": "dynamic_early_exit",
                    "confidence_threshold": confidence_threshold,
                    "num_speculations": num_speculations,
                    "acceptance_rate": metric_result['acceptance_rate']['mean'],
                    "total_time": metric_result['total_time']['mean'],
                    "time_per_token": metric_result['time_per_token']['mean'],
                    "tokens_per_second": metric_result['tokens_per_second']['mean'],
                    "avg_exit_layer": metric_result['avg_exit_layer']['mean'],
                })
                df = pd.DataFrame(results)
                df.to_csv(output_fname, index=False)
                print(f"confidence_threshold: {confidence_threshold}, num_speculations: {num_speculations}, "
                      f"avg_exit_layer: {metric_result['avg_exit_layer']['mean']}, "
                      f"time_per_token: {metric_result['time_per_token']['mean']}")
                

    elif generation_config.generation_strategy == "entropy_exit":
        # Sweep confidence threshold and num_speculations
        for entropy_threshold in np.arange(4, 8, 0.1):  
            for num_speculations in range(7, 13, 2):
                generation_config.entropy_threshold = entropy_threshold
                generation_config.num_speculations = num_speculations

                metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, args.seed)

                # print("\nFull metric_result:")
                # print(metric_result)

                # Get average exit layer from results
                # avg_exit_layer = sum(metric_result.get('exit_layers', [])) / len(metric_result.get('exit_layers', [1]))

                results.append({
                    "strategy": "dynamic_early_exit",
                    "entropy_threshold": entropy_threshold,
                    "num_speculations": num_speculations,
                    "acceptance_rate": metric_result['acceptance_rate']['mean'],
                    "total_time": metric_result['total_time']['mean'],
                    "time_per_token": metric_result['time_per_token']['mean'],
                    "tokens_per_second": metric_result['tokens_per_second']['mean'],
                    "avg_exit_layer": metric_result['avg_exit_layer']['mean'],
                    # "avg_exit_layer": avg_exit_layer
                })
                df = pd.DataFrame(results)
                df.to_csv(output_fname, index=False)
                print(f"entropy_threshold: {entropy_threshold}, num_speculations: {num_speculations}, "
                      f"avg_exit_layer: {metric_result['avg_exit_layer']['mean']}, "
                      f"time_per_token: {metric_result['time_per_token']['mean']}")


    # Print summary table
    print("\n")
    header = results[0].keys()
    rows =  [x.values() for x in results]
    print(tabulate.tabulate(rows, header))

if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    sweep(args, benchmark_arguments, generation_config, f"{args.output_dir}/sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")