import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def extract_hidden_states_after_head(model, processor, image, questions, layer_idx, head_idx, debug=False):
    """
    Extracts hidden states after a specific attention head using hooks.
    Returns representations for the last token (where prediction happens).
    
    In multi-head attention, outputs are concatenated. We extract the slice
    corresponding to the specific head by splitting the hidden dimension.
    """
    hidden_states = []
    
    # Get model config to determine head dimensions
    text_config = getattr(model.config, 'text_config', None)
    if text_config is not None:
        num_heads = text_config.num_attention_heads
        hidden_size = text_config.hidden_size
    else:
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
            num_heads = model.language_model.config.num_attention_heads
            hidden_size = model.language_model.config.hidden_size
        else:
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
    
    # Initial head dimensions based on config (may be adjusted based on actual tensor size)
    head_dim = hidden_size // num_heads
    head_start = head_idx * head_dim
    head_end = (head_idx + 1) * head_dim
    
    if debug and layer_idx == 0 and head_idx == 0:
        print(f"\n[DEBUG] Head Extraction Configuration:")
        print(f"  num_heads: {num_heads}, hidden_size: {hidden_size}")
        print(f"  head_dim: {head_dim}, head_start: {head_start}, head_end: {head_end}")
        print(f"  For head {head_idx}: extracting dimensions [{head_start}:{head_end}]")
    
    # Validate head dimensions
    if head_dim == 0:
        raise ValueError(f"Invalid head_dim: {head_dim} (hidden_size={hidden_size}, num_heads={num_heads})")
    
    # Track actual dimensions detected from tensor (will be set on first forward pass)
    actual_hidden_size = None
    actual_head_dim = None
    actual_head_start = None
    actual_head_end = None
    
    # Track which question we're processing
    question_idx = [0]  # Use list to allow modification in closure
    
    def extract_hook(module, input, output):
        nonlocal actual_hidden_size, actual_head_dim, actual_head_start, actual_head_end
        nonlocal question_idx
        
        # For layer outputs, output is typically a tuple: (hidden_states, ...)
        # For attention modules, the output structure may differ
        # We need hidden states with shape (batch, seq_len, hidden_dim)
        if isinstance(output, tuple):
            # Try first element (most common for layer outputs)
            hidden = output[0]
        else:
            hidden = output
        
        # Validate shape
        if not isinstance(hidden, torch.Tensor):
            print(f"Warning: Output is not a tensor, got {type(hidden)}")
            if actual_head_dim is not None:
                hidden_states.append(np.zeros(actual_head_dim))
            else:
                hidden_states.append(np.zeros(head_dim))
            return
        
        if len(hidden.shape) < 3:
            print(f"Warning: Unexpected hidden shape: {hidden.shape}, expected (batch, seq_len, hidden_dim)")
            if actual_head_dim is not None:
                hidden_states.append(np.zeros(actual_head_dim))
            else:
                hidden_states.append(np.zeros(head_dim))
            return
        
        # Detect actual hidden size from tensor and recalculate head dimensions
        # This handles cases where the layer output has different hidden_size than config
        if actual_hidden_size is None:
            actual_hidden_size = hidden.shape[2]
            actual_head_dim = actual_hidden_size // num_heads
            actual_head_start = head_idx * actual_head_dim
            actual_head_end = (head_idx + 1) * actual_head_dim
            
            if debug and layer_idx == 0 and head_idx == 0:
                print(f"\n[DEBUG] Actual Tensor Dimensions:")
                print(f"  hidden.shape: {hidden.shape}")
                print(f"  actual_hidden_size: {actual_hidden_size}")
                print(f"  actual_head_dim: {actual_head_dim}")
                print(f"  actual_head_start: {actual_head_start}, actual_head_end: {actual_head_end}")
                print(f"  Hooked into module: {target_module_name}")
                # Verify we're in language model, not vision tower
                if "vision" in target_module_name.lower() or "vision_tower" in target_module_name.lower():
                    print(f"  [ERROR] Hook is in VISION TOWER, not language model!")
                    print(f"  This will give identical representations across questions!")
                elif "language_model" in target_module_name.lower():
                    print(f"  [GOOD] Hook is in LANGUAGE MODEL - correct location!")
                else:
                    print(f"  [WARNING] Hook location unclear - verify it's in language model")
            
            # Validate
            if actual_head_dim == 0:
                print(f"Warning: Calculated head_dim is 0 (actual_hidden_size={actual_hidden_size}, num_heads={num_heads})")
                hidden_states.append(np.zeros(1))
                return
        
        # Check if we have enough dimensions
        if hidden.shape[2] < actual_head_end:
            print(f"Warning: Hidden dimension {hidden.shape[2]} < head_end {actual_head_end}")
            hidden_states.append(np.zeros(actual_head_dim))
            return
        
        # Extract the slice corresponding to the specific head
        # Shape: (batch, seq_len, head_dim)
        head_specific_hidden = hidden[:, :, actual_head_start:actual_head_end]
        
        # Get the last token's representation (where we predict)
        if head_specific_hidden.shape[1] == 0:
            print(f"Warning: Empty sequence length")
            hidden_states.append(np.zeros(actual_head_dim))
            return
            
        last_token_hidden = head_specific_hidden[0, -1, :].detach().cpu().numpy()
        
        # Debug: Print statistics for first few examples
        current_q_idx = question_idx[0]
        if debug and layer_idx == 0 and head_idx == 0 and current_q_idx < 3:
            print(f"\n[DEBUG] Head {head_idx} - Example {current_q_idx}:")
            print(f"  head_specific_hidden.shape: {head_specific_hidden.shape}")
            print(f"  Sequence length: {head_specific_hidden.shape[1]}")
            print(f"  Capturing token at position: {head_specific_hidden.shape[1] - 1} (last token - generation position)")
            print(f"  last_token_hidden.shape: {last_token_hidden.shape}")
            print(f"  last_token_hidden mean: {last_token_hidden.mean():.6f}, std: {last_token_hidden.std():.6f}")
            print(f"  last_token_hidden min: {last_token_hidden.min():.6f}, max: {last_token_hidden.max():.6f}")
            print(f"  First 5 values: {last_token_hidden[:5]}")
            # Check if this is different from previous example
            if current_q_idx > 0 and len(hidden_states) > 0:
                prev_hidden = hidden_states[-1]
                diff = np.abs(last_token_hidden - prev_hidden)
                print(f"  Difference from previous example - mean: {diff.mean():.6f}, max: {diff.max():.6f}")
                print(f"  Are they identical? {np.allclose(last_token_hidden, prev_hidden, atol=1e-6)}")
                if np.allclose(last_token_hidden, prev_hidden, atol=1e-6):
                    print(f"  [WARNING] This example is IDENTICAL to the previous one!")
                else:
                    print(f"  [GOOD] This example differs from the previous one!")
        
        # Validate extracted representation
        if last_token_hidden.shape[0] != actual_head_dim:
            print(f"Warning: Extracted representation has wrong size: {last_token_hidden.shape[0]} != {actual_head_dim}")
            hidden_states.append(np.zeros(actual_head_dim))
        else:
            hidden_states.append(last_token_hidden)
    
    # Find the correct module for LLaVA models
    # CRITICAL: We need to hook into the LANGUAGE MODEL, not the vision tower
    # Based on debug output, the actual path is: model.language_model.layers.{layer_idx}
    # This is the LLaMA decoder layer, not the CLIP vision encoder
    handle = None
    target_module_name = None
    
    # #region agent log
    import json
    with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"A","location":"probing.py:162","message":"Starting hook search","data":{"layer_idx":layer_idx,"head_idx":head_idx},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Priority 1: Hook into the full language model layer (after attention + MLP + residual)
    # Try: model.language_model.layers.{layer_idx} (actual structure from debug output)
    target_layer_name = f"model.language_model.layers.{layer_idx}"
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_module_name = name
            handle = module.register_forward_hook(extract_hook)
            # #region agent log
            with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"A","location":"probing.py:172","message":"Found language model layer","data":{"target_name":target_layer_name,"actual_name":name,"module_type":type(module).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            if debug and layer_idx == 0 and head_idx == 0:
                print(f"\n[DEBUG] Hook Placement:")
                print(f"  Successfully hooked into: {target_module_name}")
                print(f"  Module type: {type(module).__name__}")
            break
    
    # Priority 2: If that doesn't work, try hooking into self_attn within language model
    if handle is None:
        target_attn_name = f"model.language_model.layers.{layer_idx}.self_attn"
        for name, module in model.named_modules():
            if name == target_attn_name:
                target_module_name = name
                handle = module.register_forward_hook(extract_hook)
                # #region agent log
                with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"B","location":"probing.py:185","message":"Found language model attention","data":{"target_name":target_attn_name,"actual_name":name,"module_type":type(module).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                if debug and layer_idx == 0 and head_idx == 0:
                    print(f"\n[DEBUG] Hook Placement:")
                    print(f"  Hooked into attention module: {target_module_name}")
                    print(f"  Module type: {type(module).__name__}")
                break
    
    # Priority 3: Fallback - try alternative naming patterns
    if handle is None:
        # Try: language_model.model.layers.{layer_idx} (old pattern)
        target_layer_name_alt1 = f"language_model.model.layers.{layer_idx}"
        for name, module in model.named_modules():
            if name == target_layer_name_alt1:
                target_module_name = name
                handle = module.register_forward_hook(extract_hook)
                # #region agent log
                with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"C","location":"probing.py:200","message":"Found with alt pattern 1","data":{"target_name":target_layer_name_alt1,"actual_name":name},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                if debug and layer_idx == 0 and head_idx == 0:
                    print(f"\n[DEBUG] Hook Placement (fallback 1):")
                    print(f"  Hooked into: {target_module_name}")
                break
    
    # Priority 4: Try without "model" prefix
    if handle is None:
        target_layer_name_alt2 = f"language_model.layers.{layer_idx}"
        for name, module in model.named_modules():
            if name == target_layer_name_alt2:
                target_module_name = name
                handle = module.register_forward_hook(extract_hook)
                # #region agent log
                with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"D","location":"probing.py:215","message":"Found with alt pattern 2","data":{"target_name":target_layer_name_alt2,"actual_name":name},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                if debug and layer_idx == 0 and head_idx == 0:
                    print(f"\n[DEBUG] Hook Placement (fallback 2):")
                    print(f"  Hooked into: {target_module_name}")
                break
    
    if handle is None:
        # Debug: Print available module names to help diagnose
        available_modules = []
        for name, module in model.named_modules():
            if f"layers.{layer_idx}" in name:
                available_modules.append((name, type(module).__name__))
        # #region agent log
        with open('/home/tenkhtuvshin/DeepLearningProject/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"hook-search","hypothesisId":"E","location":"probing.py:225","message":"Hook search failed","data":{"layer_idx":layer_idx,"available_modules":available_modules},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        if debug and layer_idx == 0 and head_idx == 0:
            print(f"\n[DEBUG] Available modules with 'layers.{layer_idx}':")
            for name, mod_type in available_modules:
                print(f"  {name} (type: {mod_type})")
        raise ValueError(
            f"Could not find language model layer {layer_idx}. "
            f"Tried: 'model.language_model.layers.{layer_idx}', 'language_model.model.layers.{layer_idx}', 'language_model.layers.{layer_idx}'. "
            f"Available modules: {[m[0] for m in available_modules[:5]]}"
        )
    
    # Run forward pass for all questions
    # We generate one token to capture representation at the generation position
    # (after question is fully processed)
    for qa_item in questions:
        question_text = qa_item['question']
        prompt_text = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        # Move inputs to device properly - handle both dict and BatchFeature types
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate one token to get representation at generation position
        # This ensures we capture the representation AFTER the question is processed
        # The hook will fire during generation, capturing hidden states at the answer position
        with torch.no_grad():
            try:
                # Generate one token - hook will capture representation during generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=False
                )
            except TypeError:
                # Some models don't support use_cache parameter
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
        
        # Increment question index for debugging
        question_idx[0] += 1
    
    if handle:
        handle.remove()
    
    # Validate results
    if len(hidden_states) == 0:
        raise ValueError(f"No hidden states extracted for layer {layer_idx}, head {head_idx}")
    
    if len(hidden_states) != len(questions):
        print(f"Warning: Expected {len(questions)} hidden states, got {len(hidden_states)}")
        print(f"  This suggests the hook may not be firing for all forward passes!")
    
    result = np.array(hidden_states)
    
    # Debug: Print summary statistics
    if debug and layer_idx == 0:
        print(f"\n[DEBUG] Head {head_idx} - Summary Statistics:")
        print(f"  result.shape: {result.shape}")
        print(f"  result mean: {result.mean():.6f}, std: {result.std():.6f}")
        print(f"  result min: {result.min():.6f}, max: {result.max():.6f}")
        print(f"  Per-example means: {result.mean(axis=1)[:5]}")  # First 5 examples
        # Check if all examples are identical
        if len(result) > 1:
            first_example = result[0]
            all_same = all(np.allclose(result[i], first_example, atol=1e-6) for i in range(1, len(result)))
            if all_same:
                print(f"  [WARNING] All examples have identical representations!")
                print(f"  This means the hook is capturing the same value for all questions.")
                print(f"  Possible causes:")
                print(f"    1. Hook is placed at wrong location (not seeing question-dependent processing)")
                print(f"    2. Model is not processing questions differently")
                print(f"    3. Hook is capturing cached/constant values")
            else:
                # Show differences between first two examples
                diff = np.abs(result[1] - result[0])
                print(f"  Difference between example 0 and 1 - mean: {diff.mean():.6f}, max: {diff.max():.6f}")
                # Check if differences are meaningful
                if diff.mean() < 1e-5:
                    print(f"  [WARNING] Differences are very small - representations are nearly identical!")
        # Check if all examples are identical
        if len(result) > 1:
            first_example = result[0]
            all_same = all(np.allclose(result[i], first_example, atol=1e-6) for i in range(1, len(result)))
            if all_same:
                print(f"  [WARNING] All examples have identical representations!")
                print(f"  This means the hook is capturing the same value for all questions.")
                print(f"  Possible causes:")
                print(f"    1. Hook is placed at wrong location (not seeing question-dependent processing)")
                print(f"    2. Model is not processing questions differently")
                print(f"    3. Hook is capturing cached/constant values")
            else:
                # Show differences between first two examples
                diff = np.abs(result[1] - result[0])
                print(f"  Difference between example 0 and 1 - mean: {diff.mean():.6f}, max: {diff.max():.6f}")
                # Check if differences are meaningful
                if diff.mean() < 1e-5:
                    print(f"  [WARNING] Differences are very small - representations are nearly identical!")
    
    # Check if we got valid features
    if result.shape[1] == 0:
        raise ValueError(
            f"Extracted hidden states have 0 features for layer {layer_idx}, head {head_idx}. "
            f"Result shape: {result.shape}, actual_head_dim: {actual_head_dim}, actual_head_start: {actual_head_start}, actual_head_end: {actual_head_end}. "
            f"Hooked into: {target_module_name}"
        )
    
    return result

def probe_head_representations(model, processor, image, questions, results, layer_idx, head_idx, debug=False):
    """
    Trains a probe on representations from a specific head to predict correctness.
    Returns probe accuracy.
    """
    # Extract representations
    X = extract_hidden_states_after_head(model, processor, image, questions, layer_idx, head_idx, debug=debug)
    
    # Labels: 1 if correct, 0 if incorrect
    y = np.array([1 if r['is_correct'] else 0 for r in results])
    
    if debug:
        print(f"\n[DEBUG] Probe Training - Layer {layer_idx}, Head {head_idx}:")
        print(f"  X.shape: {X.shape}")
        print(f"  y distribution: {np.bincount(y)} (0s: {np.sum(y==0)}, 1s: {np.sum(y==1)})")
        print(f"  X mean across all examples: {X.mean():.6f}, std: {X.std():.6f}")
        print(f"  X unique values count: {len(np.unique(X.flatten()))}")
        print(f"  X sample (first 10 values): {X[0][:10] if len(X) > 0 else 'N/A'}")
    
    if len(np.unique(y)) < 2:  # Need both classes
        if debug:
            print(f"  [WARNING] Only one class in labels, returning 0.0")
        return 0.0
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if debug:
        print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"  Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
    
    # Train probe (simple logistic regression)
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if debug:
        print(f"  Accuracy: {accuracy:.6f}")
        print(f"  Predictions: {y_pred}, True: {y_test}")
        print(f"  Probe coefficients mean: {probe.coef_.mean():.6f}, std: {probe.coef_.std():.6f}")
    
    return accuracy

def compute_head_probing_scores(model, processor, image, questions, results, debug=False, single_layer=None, single_head=None):
    """
    Computes probing accuracy for each head to identify which heads encode
    information relevant to correctness.
    
    Args:
        single_layer: If specified, only probe this layer (all heads)
        single_head: If specified with single_layer, only probe this specific head
    """
    # For LLaVA models, access language model config via text_config
    text_config = getattr(model.config, 'text_config', None)
    if text_config is not None:
        num_layers = text_config.num_hidden_layers
        num_heads = text_config.num_attention_heads
    else:
        # Fallback: try language_model.config or direct access
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
            num_layers = model.language_model.config.num_hidden_layers
            num_heads = model.language_model.config.num_attention_heads
        else:
            num_layers = model.config.num_hidden_layers
            num_heads = model.config.num_attention_heads
    
    # Validate single head/layer arguments
    if single_layer is not None:
        if single_layer < 0 or single_layer >= num_layers:
            raise ValueError(f"single_layer must be between 0 and {num_layers-1}, got {single_layer}")
        if single_head is not None:
            if single_head < 0 or single_head >= num_heads:
                raise ValueError(f"single_head must be between 0 and {num_heads-1}, got {single_head}")
    
    if debug:
        print(f"\n[DEBUG] Model Configuration:")
        print(f"  num_layers: {num_layers}, num_heads: {num_heads}")
        if single_layer is not None:
            if single_head is not None:
                print(f"  Probing: Layer {single_layer}, Head {single_head} only")
            else:
                print(f"  Probing: Layer {single_layer} only (all {num_heads} heads)")
        else:
            print(f"  Total heads to probe: {num_layers * num_heads}")
        print(f"  Number of questions: {len(questions)}")
        print(f"  Number of results: {len(results)}")
    
    probing_scores = {}  # {(layer, head): accuracy}
    
    # Store representations for comparison (debug only)
    head_representations_cache = {} if debug else None
    
    # Determine which layers/heads to probe
    if single_layer is not None:
        layers_to_probe = [single_layer]
    else:
        layers_to_probe = range(num_layers)
    
    if single_head is not None:
        heads_to_probe = [single_head]
    else:
        heads_to_probe = range(num_heads)
    
    print(f"\nComputing head-level probing scores...")
    if single_layer is not None and single_head is not None:
        print(f"Probing Layer {single_layer}, Head {single_head} only")
    elif single_layer is not None:
        print(f"Probing Layer {single_layer} only (all {num_heads} heads)")
    
    for layer_idx in layers_to_probe:
        if debug:
            print(f"\n[DEBUG] Processing Layer {layer_idx}...")
        
        for head_idx in heads_to_probe:
            # Enable full debug for single head probing
            enable_debug = debug or (single_layer is not None and single_head is not None)
            accuracy = probe_head_representations(
                model, processor, image, questions, results, layer_idx, head_idx, debug=enable_debug
            )
            probing_scores[(layer_idx, head_idx)] = accuracy
            
            if single_layer is not None and single_head is not None:
                print(f"\nResult: Layer {layer_idx}, Head {head_idx} accuracy = {accuracy:.6f}")
            
            # Debug: Compare first few heads to check if they're different
            if debug and layer_idx == 0 and head_idx < 3:
                if head_representations_cache is not None:
                    head_representations_cache[head_idx] = accuracy
                    if head_idx > 0:
                        # Compare accuracies (if they're all the same, heads might be extracting same data)
                        print(f"\n[DEBUG] Head {head_idx} vs Head 0 accuracy comparison:")
                        print(f"  Head 0 accuracy: {head_representations_cache[0]:.6f}")
                        print(f"  Head {head_idx} accuracy: {accuracy:.6f}")
                        print(f"  Difference: {abs(accuracy - head_representations_cache[0]):.6f}")
    
    if debug or (single_layer is not None and single_head is not None):
        # Check if all scores are the same
        scores = list(probing_scores.values())
        unique_scores = np.unique(scores)
        print(f"\n[DEBUG] Probing Scores Summary:")
        print(f"  Unique accuracy values: {len(unique_scores)}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"  Mean score: {np.mean(scores):.6f}, Std: {np.std(scores):.6f}")
        if len(unique_scores) == 1:
            print(f"  [WARNING] All scores are identical: {unique_scores[0]}")
        if len(scores) <= 10:
            print(f"  All scores: {scores}")
        else:
            print(f"  First 10 scores: {scores[:10]}")
    
    return probing_scores


def extract_hidden_states_after_layer(model, processor, image, questions, layer_idx, debug=False):
    """
    Extracts hidden states after a specific layer (residual stream after attention + MLP).
    Returns representations at generation-step-1 token position (where answer is generated).
    
    This captures the full 4096-dim residual stream, not head-specific slices.
    """
    hidden_states = []
    target_module_name = None
    
    # Track which question we're processing
    question_idx = [0]
    
    def extract_hook(module, input, output):
        nonlocal question_idx
        
        # output is typically a tuple: (hidden_states, ...) for decoder layers
        if isinstance(output, tuple):
            hidden = output[0]  # (batch, seq_len, hidden_dim)
        else:
            hidden = output
        
        # Validate shape
        if not isinstance(hidden, torch.Tensor):
            print(f"Warning: Output is not a tensor, got {type(hidden)}")
            return
        
        if len(hidden.shape) < 3:
            print(f"Warning: Unexpected hidden shape: {hidden.shape}, expected (batch, seq_len, hidden_dim)")
            return
        
        # Get the last token's representation (generation position)
        # This is where the model generates the answer
        last_token_hidden = hidden[0, -1, :].detach().cpu().numpy()
        
        current_q_idx = question_idx[0]
        if debug and layer_idx == 0 and current_q_idx < 3:
            print(f"\n[DEBUG] Layer {layer_idx} - Example {current_q_idx}:")
            print(f"  hidden.shape: {hidden.shape}")
            print(f"  Capturing token at position: {hidden.shape[1] - 1} (generation position)")
            print(f"  last_token_hidden.shape: {last_token_hidden.shape}")
            print(f"  last_token_hidden mean: {last_token_hidden.mean():.6f}, std: {last_token_hidden.std():.6f}")
            # Check if this differs from previous example
            if current_q_idx > 0 and len(hidden_states) > 0:
                prev_hidden = hidden_states[-1]
                diff = np.abs(last_token_hidden - prev_hidden)
                print(f"  Difference from previous - mean: {diff.mean():.6f}, max: {diff.max():.6f}")
                if np.allclose(last_token_hidden, prev_hidden, atol=1e-6):
                    print(f"  [WARNING] Identical to previous example!")
                else:
                    print(f"  [GOOD] Differs from previous example!")
        
        hidden_states.append(last_token_hidden)
    
    # CRITICAL: Hook into LANGUAGE MODEL layer, not vision tower
    # Target: model.language_model.layers.{layer_idx} (residual stream after full layer)
    handle = None
    
    # Priority 1: Hook into the full language model layer (after attention + MLP + residual)
    target_layer_name = f"model.language_model.layers.{layer_idx}"
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_module_name = name
            handle = module.register_forward_hook(extract_hook)
            if debug and layer_idx == 0:
                print(f"\n[DEBUG] Layer Hook Placement:")
                print(f"  Successfully hooked into: {target_module_name}")
                print(f"  Module type: {type(module).__name__}")
                if "vision" in target_module_name.lower():
                    print(f"  [ERROR] Hook is in VISION TOWER, not language model!")
                elif "language_model" in target_module_name.lower():
                    print(f"  [GOOD] Hook is in LANGUAGE MODEL - correct location!")
            break
    
    # Priority 2: Fallback - try alternative naming
    if handle is None:
        target_layer_name_alt = f"language_model.model.layers.{layer_idx}"
        for name, module in model.named_modules():
            if name == target_layer_name_alt:
                target_module_name = name
                handle = module.register_forward_hook(extract_hook)
                if debug and layer_idx == 0:
                    print(f"\n[DEBUG] Layer Hook Placement (fallback):")
                    print(f"  Hooked into: {target_module_name}")
                break
    
    if handle is None:
        raise ValueError(
            f"Could not find language model layer {layer_idx}. "
            f"Tried: 'model.language_model.layers.{layer_idx}', 'language_model.model.layers.{layer_idx}'"
        )
    
    # Run forward pass for all questions
    # Use generation to capture at generation-step-1 position
    for qa_item in questions:
        question_text = qa_item['question']
        prompt_text = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        # Move inputs to device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate one token to capture representation at generation position
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=False
                )
            except TypeError:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
        
        question_idx[0] += 1
    
    if handle:
        handle.remove()
    
    # Validate results
    if len(hidden_states) == 0:
        raise ValueError(f"No hidden states extracted for layer {layer_idx}")
    
    if len(hidden_states) != len(questions):
        print(f"Warning: Expected {len(questions)} hidden states, got {len(hidden_states)}")
    
    result = np.array(hidden_states)
    
    # Debug: Check if all examples are identical
    if debug and layer_idx == 0:
        if len(result) > 1:
            first_example = result[0]
            all_same = all(np.allclose(result[i], first_example, atol=1e-6) for i in range(1, len(result)))
            if all_same:
                print(f"\n[DEBUG] Layer {layer_idx} - All examples have identical representations!")
            else:
                diff = np.abs(result[1] - result[0])
                print(f"\n[DEBUG] Layer {layer_idx} - Difference between examples:")
                print(f"  Mean diff: {diff.mean():.6f}, Max diff: {diff.max():.6f}")
                print(f"  Result shape: {result.shape}, Expected: (num_questions, 4096)")
    
    return result


def probe_layer_representations(model, processor, image, questions, results, layer_idx, debug=False):
    """
    Trains a probe on representations from a specific layer to predict correctness.
    Returns probe accuracy.
    
    This probes the residual stream (4096 dims) at generation-step-1 position.
    """
    # Extract representations
    X = extract_hidden_states_after_layer(model, processor, image, questions, layer_idx, debug=debug)
    
    # Labels: 1 if correct, 0 if incorrect
    y = np.array([1 if r['is_correct'] else 0 for r in results])
    
    if debug and layer_idx == 0:
        print(f"\n[DEBUG] Layer {layer_idx} Probe Training:")
        print(f"  X.shape: {X.shape} (expected: (num_questions, 4096))")
        print(f"  y distribution: {np.bincount(y)} (0s: {np.sum(y==0)}, 1s: {np.sum(y==1)})")
        print(f"  X mean: {X.mean():.6f}, std: {X.std():.6f}")
    
    if len(np.unique(y)) < 2:  # Need both classes
        if debug:
            print(f"  [WARNING] Only one class in labels, returning 0.0")
        return 0.0
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if debug and layer_idx == 0:
        print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train probe (linear probe - logistic regression)
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if debug and layer_idx == 0:
        print(f"  Accuracy: {accuracy:.6f}")
        print(f"  Predictions: {y_pred}, True: {y_test}")
    
    return accuracy


def compute_layer_probing_scores(model, processor, image, questions, results, debug=False):
    """
    Computes probing accuracy for each layer to identify which layers encode
    information relevant to correctness.
    
    Probes the residual stream (4096 dims) at generation-step-1 position.
    """
    # For LLaVA models, access language model config via text_config
    text_config = getattr(model.config, 'text_config', None)
    if text_config is not None:
        num_layers = text_config.num_hidden_layers
    else:
        # Fallback: try language_model.config or direct access
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
            num_layers = model.language_model.config.num_hidden_layers
        else:
            num_layers = model.config.num_hidden_layers
    
    if debug:
        print(f"\n[DEBUG] Layer Probing Configuration:")
        print(f"  num_layers: {num_layers}")
        print(f"  Number of questions: {len(questions)}")
        print(f"  Number of results: {len(results)}")
        print(f"  Probing residual stream (4096 dims) at generation-step-1 position")
    
    probing_scores = {}  # {layer: accuracy}
    
    print("\nComputing layer-level probing scores...")
    print("Probing residual stream after each layer (4096 dims)")
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        # Enable debug for first layer only
        enable_debug = debug and layer_idx == 0
        accuracy = probe_layer_representations(
            model, processor, image, questions, results, layer_idx, debug=enable_debug
        )
        probing_scores[layer_idx] = accuracy
        
        if debug and layer_idx == 0:
            print(f"\nLayer {layer_idx} accuracy: {accuracy:.6f}")
    
    if debug:
        scores = list(probing_scores.values())
        print(f"\n[DEBUG] Layer Probing Summary:")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"  Mean score: {np.mean(scores):.6f}, Std: {np.std(scores):.6f}")
        print(f"  Top 5 layers: {sorted(probing_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return probing_scores


def plot_head_probing_heatmap(probing_scores, output_dir, title_id):
    """
    Creates a heatmap showing probing accuracy for each head across all layers.
    """
    if not probing_scores:
        return
    
    # Get dimensions
    layers = sorted(set([k[0] for k in probing_scores.keys()]))
    heads = sorted(set([k[1] for k in probing_scores.keys()]))
    num_layers = len(layers)
    num_heads = len(heads)
    
    # Build matrix
    probing_matrix = np.zeros((num_layers, num_heads))
    for (layer_idx, head_idx), accuracy in probing_scores.items():
        if layer_idx in layers and head_idx in heads:
            row = layers.index(layer_idx)
            col = heads.index(head_idx)
            probing_matrix[row, col] = accuracy
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(probing_matrix, aspect='auto', cmap='YlGnBu', interpolation='nearest', vmin=0.0, vmax=1.0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probe Accuracy', fontsize=12, fontweight='bold')
    
    # Set labels
    ax.set_xlabel('Attention Head Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_title(f'Head-Level Probing Results: {title_id}\n(Higher = More Information About Correctness)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(heads)
    ax.set_yticklabels(layers)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Highlight top heads (top 10% by average across all layers)
    head_avg = np.mean(probing_matrix, axis=0)
    if len(head_avg) > 0:
        top_head_indices = np.argsort(head_avg)[-max(1, num_heads//10):][::-1]
        for head_idx in top_head_indices:
            ax.axvline(x=head_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "head_probing_heatmap.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Head probing heatmap saved to: {save_path}")
    plt.close()


def plot_layer_probing_scores(probing_scores, output_dir, title_id):
    """
    Creates a line plot showing probing accuracy across layers.
    """
    if not probing_scores:
        return
    
    layers = sorted(probing_scores.keys())
    accuracies = [probing_scores[l] for l in layers]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Layer Index', fontsize=12, fontweight='bold')
    plt.ylabel('Probe Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Layer-Level Probing Results: {title_id}\n(Higher = More Information About Correctness)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.0)
    
    # Highlight top layers
    top_indices = np.argsort(accuracies)[-5:][::-1]
    for idx in top_indices:
        plt.scatter(layers[idx], accuracies[idx], s=200, color='red', marker='*', zorder=5)
        plt.annotate(f'L{layers[idx]}\n{accuracies[idx]:.3f}', 
                    xy=(layers[idx], accuracies[idx]),
                    xytext=(5, 15), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "layer_probing_scores.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Layer probing plot saved to: {save_path}")
    plt.close()