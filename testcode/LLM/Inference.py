import argparse
import os
import numpy as np
import time
import tvm
from transformers import AutoTokenizer, AutoConfig

from gpt2_dtype_utils import (
    convert_float_array_to_runtime_tensor,
    convert_runtime_tensor_to_float_numpy,
    register_custom_datatypes,
    validate_dtype_arg,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
TAG = "tvm"

processor = AutoTokenizer.from_pretrained("onnx-community/gpt2-ONNX")
config = AutoConfig.from_pretrained("onnx-community/gpt2-ONNX")

def calculate_perplexity(logits_list, target_tokens):
    """Calculate perplexity from logits and target tokens

    Args:
        logits_list: List of logit arrays (vocab_size,) for each step
        target_tokens: List of actual next tokens

    Returns:
        perplexity: Float value
    """
    if len(logits_list) != len(target_tokens):
        print(f"Warning: logits_list length {len(logits_list)} != target_tokens length {len(target_tokens)}")
        min_len = min(len(logits_list), len(target_tokens))
        logits_list = logits_list[:min_len]
        target_tokens = target_tokens[:min_len]

    log_likelihoods = []
    for logits, target_token in zip(logits_list, target_tokens):
        # Apply softmax to get probabilities
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Get probability of the target token
        target_prob = probs[target_token]

        # Avoid log(0)
        if target_prob <= 0:
            target_prob = 1e-10

        log_likelihood = np.log(target_prob)
        log_likelihoods.append(log_likelihood)

    # Calculate perplexity
    avg_log_likelihood = np.mean(log_likelihoods)
    perplexity = np.exp(-avg_log_likelihood)

    return perplexity

def _parse_teacher_tokens_arg(value):
    if value is None:
        return None
    tokens = []
    for idx, item in enumerate(value.split(",")):
        item = item.strip()
        if not item:
            continue
        try:
            token = int(item)
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                f"Invalid token '{item}' at position {idx} in --teacher-tokens"
            ) from err
        if token < 0:
            raise argparse.ArgumentTypeError(
                f"Invalid token '{item}' at position {idx}: token id must be >= 0"
            )
        tokens.append(token)
    if not tokens:
        raise argparse.ArgumentTypeError("--teacher-tokens provided but no valid token ids found")
    return tokens


def benchmark_dual_models(
    model1_path,
    model2_path,
    dtype1,
    dtype2,
    input_data,
    decode_mode="greedy",
    teacher_tokens=None,
):
    """
    Benchmark two models with different data types
    dtype1, dtype2: float16/float32 or custom[name]bits
    """
    if decode_mode == "teacher":
        if teacher_tokens is None or len(teacher_tokens) == 0:
            raise ValueError("decode_mode='teacher' requires non-empty teacher_tokens")

    # Load both models
    lib_model1 = tvm.runtime.load_module(model1_path)
    decoder_model1 = tvm.relax.vm.VirtualMachine(lib_model1, tvm.device("cpu"))

    lib_model2 = tvm.runtime.load_module(model2_path)
    decoder_model2 = tvm.relax.vm.VirtualMachine(lib_model2, tvm.device("cpu"))

    # Extract parameters
    batch_size = 1
    num_layers = config.n_layer
    num_key_value_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    device = tvm.device("cpu")
    # Initialize KV caches for both models
    kvcache_model1 = [
        convert_float_array_to_runtime_tensor(
            np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32),
            dtype1,
            device,
        )
        for _ in range(num_layers * 2)
    ]
    kvcache_model2 = [
        convert_float_array_to_runtime_tensor(
            np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32),
            dtype2,
            device,
        )
        for _ in range(num_layers * 2)
    ]

    input_ids = processor(input_data).input_ids
    attention_mask = [1]
    position_ids = [0]

    # Performance tracking
    prefill_time_model1 = 0
    prefill_time_model2 = 0
    decode_times_model1 = []
    decode_times_model2 = []
    total_generation_time_model1 = 0
    total_generation_time_model2 = 0

    # Error tracking
    logit_errors = []  # MAE between logits
    token_matches = []  # Whether tokens match
    generation_tokens_model1 = []
    generation_tokens_model2 = []

    # Perplexity tracking
    logits_history_model1 = []  # Store all logits for perplexity calculation
    logits_history_model2 = []
    teacher_targets = None

    # Prefill phase for both models
    # print("\n" + "="*80)
    # print("PREFILL PHASE")
    # print("="*80)

    # Model 1 prefill
    prefill_start_model1 = time.time()
    for input_id in input_ids:
        out_model1 = decoder_model1["main"](
            tvm.runtime.tensor([[input_id]]),
            *kvcache_model1,
            tvm.runtime.tensor([attention_mask]),
            tvm.runtime.tensor([position_ids])
        )
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache_model1 = out_model1[1:]
    prefill_time_model1 = time.time() - prefill_start_model1

    # Reset for model2 prefill
    position_ids = [0]
    attention_mask = [1]

    # Model 2 prefill
    prefill_start_model2 = time.time()
    for input_id in input_ids:
        out_model2 = decoder_model2["main"](
            tvm.runtime.tensor([[input_id]]),
            *kvcache_model2,
            tvm.runtime.tensor([attention_mask]),
            tvm.runtime.tensor([position_ids])
        )
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache_model2 = out_model2[1:]
    prefill_time_model2 = time.time() - prefill_start_model2

    # Get first token from both models
    logits_model1_raw = out_model1[0]
    logits_model1 = convert_runtime_tensor_to_float_numpy(
        logits_model1_raw, dtype1, device
    )[0][-1]

    logits_model2_raw = out_model2[0]
    logits_model2 = convert_runtime_tensor_to_float_numpy(
        logits_model2_raw, dtype2, device
    )[0][-1]

    # Store logits for perplexity
    logits_history_model1.append(logits_model1.copy())
    logits_history_model2.append(logits_model2.copy())

    # Calculate first logit error
    mae = np.mean(np.abs(logits_model1 - logits_model2))
    logit_errors.append(mae)

    # print(f"Prefill Logits MAE: {mae:.6f}")

    next_token_model1 = np.argmax(logits_model1, axis=-1)
    next_token_model2 = np.argmax(logits_model2, axis=-1)

    if decode_mode == "teacher":
        effective_iter = min(ITER, len(teacher_tokens))
        teacher_targets = teacher_tokens[:effective_iter]
        if effective_iter < ITER:
            print(
                f"Warning: teacher token path shorter than --iter ({len(teacher_tokens)} < {ITER}). "
                f"Using {effective_iter} steps."
            )
    else:
        effective_iter = ITER

    token_match = (next_token_model1 == next_token_model2)
    token_matches.append(token_match)

    generation_tokens_model1.append(next_token_model1)
    generation_tokens_model2.append(next_token_model2)

    token_str_model1 = processor.decode([int(next_token_model1)])
    token_str_model2 = processor.decode([int(next_token_model2)])
    print("\nPrefill result:")
    print(
        f"  Model 1: Token={int(next_token_model1)}, Text={token_str_model1!r}, "
        f"Logit={logits_model1[next_token_model1]:.4f}, "
        f"Time={prefill_time_model1:.6f}s"
    )
    print(
        f"  Model 2: Token={int(next_token_model2)}, Text={token_str_model2!r}, "
        f"Logit={logits_model2[next_token_model2]:.4f}, "
        f"Time={prefill_time_model2:.6f}s"
    )
    print(f"  Logits MAE: {mae:.6f}, Token Match: {bool(token_match)}")

    # Generation loop
    print("\nDecode steps:")

    step = 1
    while len(generation_tokens_model1) < effective_iter and next_token_model1 != 151643:
        if decode_mode == "greedy":
            input_token_model1 = int(next_token_model1)
            input_token_model2 = int(next_token_model2)
            decode_driver = f"greedy(m1={input_token_model1}, m2={input_token_model2})"
        elif decode_mode == "shared":
            shared_token = int(next_token_model1)
            input_token_model1 = shared_token
            input_token_model2 = shared_token
            decode_driver = f"shared(model1={shared_token})"
        elif decode_mode == "teacher":
            teacher_feed_idx = len(generation_tokens_model1) - 1
            if teacher_feed_idx >= (effective_iter - 1):
                break
            forced_token = int(teacher_targets[teacher_feed_idx])
            input_token_model1 = forced_token
            input_token_model2 = forced_token
            decode_driver = f"teacher(token={forced_token})"
        else:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")

        input_ids_model1 = np.array([[input_token_model1]])
        input_ids_model2 = np.array([[input_token_model2]])
        tvm_inputs0_model1 = [tvm.runtime.tensor(input_ids_model1)]
        tvm_inputs1 = [tvm.runtime.tensor([attention_mask]), tvm.runtime.tensor([position_ids])]

        start = time.time()
        out_model1 = decoder_model1["main"](*tvm_inputs0_model1, *kvcache_model1, *tvm_inputs1)
        decode_time_model1 = time.time() - start
        decode_times_model1.append(decode_time_model1)

        kvcache_model1 = out_model1[1:]
        logits_model1_raw = out_model1[0]
        logits_model1 = convert_runtime_tensor_to_float_numpy(
            logits_model1_raw, dtype1, device
        )[0][-1]

        # Store logits for perplexity
        logits_history_model1.append(logits_model1.copy())

        next_token_model1 = np.argmax(logits_model1, axis=-1)
        generation_tokens_model1.append(next_token_model1)

        # Model 2
        # input_ids_model2 = next_token_model2.reshape(1, 1)
        tvm_inputs0_model2 = [tvm.runtime.tensor(input_ids_model2)]

        start = time.time()
        out_model2 = decoder_model2["main"](*tvm_inputs0_model2, *kvcache_model2, *tvm_inputs1)
        decode_time_model2 = time.time() - start
        decode_times_model2.append(decode_time_model2)

        kvcache_model2 = out_model2[1:]
        logits_model2_raw = out_model2[0]
        logits_model2 = convert_runtime_tensor_to_float_numpy(
            logits_model2_raw, dtype2, device
        )[0][-1]

        # Store logits for perplexity
        logits_history_model2.append(logits_model2.copy())

        next_token_model2 = np.argmax(logits_model2, axis=-1)
        generation_tokens_model2.append(next_token_model2)

        # Calculate error
        mae = np.mean(np.abs(logits_model1 - logits_model2))
        logit_errors.append(mae)

        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)

        token_match = (next_token_model1 == next_token_model2)
        token_matches.append(token_match)

        token_str_model1 = processor.decode([int(next_token_model1)])
        token_str_model2 = processor.decode([int(next_token_model2)])

        print(f"\nStep {step} ({decode_driver}):")
        print(
            f"  Model 1: Token={int(next_token_model1)}, Text={token_str_model1!r}, "
            f"Logit={logits_model1[next_token_model1]:.4f}, "
            f"Time={decode_time_model1:.6f}s"
        )
        print(
            f"  Model 2: Token={int(next_token_model2)}, Text={token_str_model2!r}, "
            f"Logit={logits_model2[next_token_model2]:.4f}, "
            f"Time={decode_time_model2:.6f}s"
        )
        print(f"  Logits MAE: {mae:.6f}, Token Match: {bool(token_match)}")

        step += 1

    # Calculate total generation times (prefill + decode)
    total_generation_time_model1 = prefill_time_model1 + np.sum(decode_times_model1)
    total_generation_time_model2 = prefill_time_model2 + np.sum(decode_times_model2)
    avg_decode_time_model1 = np.mean(decode_times_model1) if decode_times_model1 else 0.0
    avg_decode_time_model2 = np.mean(decode_times_model2) if decode_times_model2 else 0.0
    std_decode_time_model1 = np.std(decode_times_model1) if decode_times_model1 else 0.0
    std_decode_time_model2 = np.std(decode_times_model2) if decode_times_model2 else 0.0

    # Print profiling results
    print("\nBenchmark results:")

    print(f"\nModel 1 ({dtype1}) Performance:")
    print(f"  Prefill Time: {prefill_time_model1:.6f} seconds")
    print(f"  Avg Decode Time: {avg_decode_time_model1:.6f} ± {std_decode_time_model1:.6f} seconds")
    print(f"  Total Decode Time: {np.sum(decode_times_model1):.6f} seconds")
    print(f"  Total Generation Time: {total_generation_time_model1:.6f} seconds")
    print(f"  Avg Time per Token: {total_generation_time_model1 / len(generation_tokens_model1):.6f} seconds")

    print(f"\nModel 2 ({dtype2}) Performance:")
    print(f"  Prefill Time: {prefill_time_model2:.6f} seconds")
    print(f"  Avg Decode Time: {avg_decode_time_model2:.6f} ± {std_decode_time_model2:.6f} seconds")
    print(f"  Total Decode Time: {np.sum(decode_times_model2):.6f} seconds")
    print(f"  Total Generation Time: {total_generation_time_model2:.6f} seconds")
    print(f"  Avg Time per Token: {total_generation_time_model2 / len(generation_tokens_model2):.6f} seconds")

    perplexity_targets_model1 = generation_tokens_model1
    perplexity_targets_model2 = generation_tokens_model2
    if decode_mode == "teacher":
        # In teacher forcing, evaluate both models on the same externally provided reference path.
        perplexity_targets_model1 = teacher_targets
        perplexity_targets_model2 = teacher_targets
    perplexity_model1 = calculate_perplexity(logits_history_model1, perplexity_targets_model1)
    perplexity_model2 = calculate_perplexity(logits_history_model2, perplexity_targets_model2)

    print("\nComparison:")
    print(f"  Model 1 Perplexity: {perplexity_model1:.4f}")
    print(f"  Model 2 Perplexity: {perplexity_model2:.4f}")
    print(f"  Perplexity Difference: {abs(perplexity_model1 - perplexity_model2):.4f}")
    print(f"  Avg Logits MAE: {np.mean(logit_errors):.6f} ± {np.std(logit_errors):.6f}")
    print(f"  Max Logits MAE: {np.max(logit_errors):.6f}")
    print(f"  Min Logits MAE: {np.min(logit_errors):.6f}")
    print(f"  Token Match Rate: {np.mean(token_matches)*100:.2f}% ({np.sum(token_matches)}/{len(token_matches)})")

    # Decode final sentences
    token_ids_model1 = [int(token) for token in generation_tokens_model1]
    token_ids_model2 = [int(token) for token in generation_tokens_model2]
    output_model1 = processor.decode(token_ids_model1, skip_special_tokens=True)
    output_model2 = processor.decode(token_ids_model2, skip_special_tokens=True)

    print("\nGenerated output:")
    # print(f"  Model 1 Token IDs: {token_ids_model1}")
    # print(f"  Model 2 Token IDs: {token_ids_model2}")
    print(f"  Model 1 Text: {output_model1!r}")
    print(f"  Model 2 Text: {output_model2!r}")
    print(f"  Outputs Match: {output_model1 == output_model2}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Benchmark TVM runtime dual model comparison with flexible dtype support"
    )
    p.add_argument("--tag", required=True, help="tag for the benchmark")
    p.add_argument("--model1", required=True, help="First model .so file path")
    p.add_argument("--model2", required=True, help="Second model .so file path")
    p.add_argument("--dtype1", required=True,
                   type=validate_dtype_arg,
                   help="Data type for first model")
    p.add_argument("--dtype2", required=True,
                   type=validate_dtype_arg,
                   help="Data type for second model")
    p.add_argument("--num-threads", type=int, required=True, help="number of threads")
    p.add_argument("--is-config-threadpool", type=bool, default=False, help="config threadpool")
    p.add_argument("--iter", "-n", type=int, default=50, help="number of timed iterations (default: 50)")
    p.add_argument("--prompt", default="Tell me about AI", help="input prompt for generation")
    p.add_argument(
        "--decode-mode",
        choices=["greedy", "shared", "teacher"],
        default="shared",
        help="Decode mode: greedy (each model greedy), shared (both follow model1 greedy path), teacher (both follow external reference tokens)",
    )
    p.add_argument(
        "--teacher-text",
        default=None,
        help="Reference text used to build teacher forcing token path (only for --decode-mode teacher)",
    )
    p.add_argument(
        "--teacher-tokens",
        default=None,
        help="Comma-separated token ids for teacher forcing path (only for --decode-mode teacher)",
    )

    args = p.parse_args()
    register_custom_datatypes((args.dtype1, args.dtype2), target="llvm")
    ITER = args.iter
    TAG = args.tag
    IS_CONFIG_THREADPOOL = args.is_config_threadpool
    NUM_THREADS = args.num_threads
    decode_mode = args.decode_mode

    teacher_tokens = None
    if decode_mode == "teacher":
        if args.teacher_tokens:
            teacher_tokens = _parse_teacher_tokens_arg(args.teacher_tokens)
        elif args.teacher_text:
            teacher_tokens = processor(args.teacher_text).input_ids
        else:
            raise ValueError(
                "--decode-mode teacher requires --teacher-tokens or --teacher-text"
            )

        if len(teacher_tokens) == 0:
            raise ValueError("Teacher forcing token path is empty")

    if IS_CONFIG_THREADPOOL:
        config_func = tvm.get_global_func("runtime.config_threadpool")
        cpus = [str(i) for i in range(NUM_THREADS)]
        config_func(-2, NUM_THREADS, cpus)
    else:
        os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)

    print("\nConfiguration:")
    print(f"  Tag: {TAG}")
    print(f"  Model 1: {args.model1} ({args.dtype1})")
    print(f"  Model 2: {args.model2} ({args.dtype2})")
    print(f"  Threads: {NUM_THREADS}")
    print(f"  Iterations: {ITER}")
    print(f"  Decode Mode: {decode_mode}")
    print(f"  Prompt: {args.prompt!r}")
    if decode_mode == "teacher":
        print(f"  Teacher Tokens: {len(teacher_tokens)}")

    benchmark_dual_models(
        args.model1,
        args.model2,
        args.dtype1,
        args.dtype2,
        args.prompt,
        decode_mode=decode_mode,
        teacher_tokens=teacher_tokens,
    )
    print("\nDone!")
