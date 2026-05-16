# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import random
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
    from math_verify.grader import verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def extract_all_boxed(text: str) -> list:
    """Extract all \boxed{...} expressions with balanced brace parsing.

    Returns a list of (start_pos, end_pos, content).
    Works across newlines and nested braces.
    """
    results = []
    i = 0
    n = len(text)
    
    # Pre-compile regex for performance: \boxed followed by optional whitespace then {
    pattern = re.compile(r"\\boxed\s*\{")

    while i < n:
        match = pattern.search(text, i)
        if not match:
            break
            
        start = match.start()
        # match.end() points to char AFTER '{'
        content_start = match.end()
        
        current_idx = content_start
        depth = 1
        
        while current_idx < n and depth > 0:
            ch = text[current_idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            current_idx += 1
            
        if depth == 0:
            # Successfully paired
            content = text[content_start : current_idx - 1]
            results.append((start, current_idx, content))
            # Move i to the end of this box to extract only top-level boxes
            i = current_idx
        else:
            # Unbalanced or end of string reached without closing.
            # Skip the \boxed part and try continuing search
            i = match.end()

    return results


def _flatten_inner_boxed(content: str) -> str:
    """Remove nested \boxed{...} recursively, keeping innermost content."""
    max_iter = 10
    for _ in range(max_iter):
        boxes = extract_all_boxed(content)
        if not boxes:
            break
        # Replace the first occurrence with its content (remove the box marker)
        s, e, inner = boxes[0]
        content = content[:s] + inner + content[e:]
    return content


def preprocess_model_output(output: str) -> str:
    """Preprocess model output to handle common edge cases with Math-Verify.
    
    Common issues addressed:
    0. GSM8K style "####": Use content after the marker
    1. Nested \boxed{}: \boxed{(\boxed{1, 2})} -> \boxed{(1, 2)}
    2. Multiple \boxed{} (for MATH single-answer): Keep only the last one
    
    Args:
        output: Raw model output string
        
    Returns:
        Preprocessed output string
    """
    # Issue 0: Handle GSM8K style "####"
    if "####" in output:
        return output.split("####")[1].strip()

    # Issue 1: Handle nested \boxed{} - remove inner \boxed{} tags
    # Example: \boxed{(\boxed{1, 2})} should become \boxed{(1, 2)}
    
    max_iterations = 10
    iteration = 0
    
    original_output = output

    while iteration < max_iterations:
        prev_output = output
        output = re.sub(
            r'\\boxed\{([^{}]*?)\\boxed\{([^{}]+)\}([^{}]*?)\}',
            r'\\boxed{\1\2\3}',
            output,
            flags=re.DOTALL,
        )
        if output == prev_output:
            break
        iteration += 1

    boxed_expressions = extract_all_boxed(output)

    if boxed_expressions:
        # Keep only the last boxed; flatten any nested boxed markers inside it
        _, _, last_content = boxed_expressions[-1]
        cleaned_content = _flatten_inner_boxed(last_content).strip()
        if cleaned_content:
            return cleaned_content

    # Fallback: try to extract a plausible final expression when no \boxed{} is present.
    # Heuristics: take the last math-looking token (fraction / sqrt / number) inside the text.
    # This is a best-effort to make math_verify usable when the model omits \boxed{}.
    fallback_patterns = [
        r"\\frac\{[^{}]+\}\{[^{}]+\}",  # \frac{a}{b}
        r"\\sqrt\{[^{}]+\}",             # \sqrt{...}
        r"-?\d{1,3}(?:,\d{3})+\.?\d*",     # numbers with commas
        r"-?\d+\\.\d+",                  # decimal numbers
        r"-?\d+",                           # integers
    ]
    for pat in fallback_patterns:
        matches = re.findall(pat, output)
        if matches:
            candidate = matches[-1].strip()
            return candidate

    # Nothing found: preserve original for debugging
    # Also emit a sampled log to estimate missing-box rate.
    if random.random() < 0.001:
        preview = output[:200].replace("\n", " ")    
    return original_output


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0):
    """Compute score using Math-Verify with preprocessing to handle edge cases.
    
    Args:
        model_output: Model's output string (may contain multiple \boxed{} expressions)
        ground_truth: Ground truth answer (will be wrapped in \boxed{})
        timeout_score: Score to return if verification times out
        
    Returns:
        Score (1.0 if correct, 0.0 if incorrect)
    """
    import random

    # Preprocess model output to handle edge cases
    processed_output = preprocess_model_output(model_output)

    # Debug print: keep user-requested sampling rate (50%) for visibility while debugging
    '''if random.random() < 0.0001:
        print(
            "\n[MathVerify Debug]"
            f"\nGround Truth: {ground_truth}"
            f"\nModel Output:\n{model_output}"
            f"\nPreprocessed:\n{processed_output}\n"
        )'''

    ret_score = 0.0
    
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    canonical_pred = None
    canonical_gold = None

    try:
        # Use lower-level primitives to control timeout and threading behavior
        # Note: parsing_timeout and timeout_seconds must be None in threaded envs
        
        gold_config = [LatexExtractionConfig()]
        pred_config = [ExprExtractionConfig(), LatexExtractionConfig()]
        
        # Parse results
        extracted_gold = parse(
            ground_truth_boxed, 
            extraction_config=gold_config, 
            parsing_timeout=None
        )
        canonical_gold = str(extracted_gold[-1]) if extracted_gold else ground_truth_boxed

        extracted_pred = parse(
            processed_output, 
            extraction_config=pred_config, 
            parsing_timeout=None
        )
        canonical_pred = str(extracted_pred[-1]) if extracted_pred else processed_output
        
        # Verify
        if verify(extracted_gold, extracted_pred, timeout_seconds=None):
            ret_score = 1.0
        else:
            ret_score = 0.0
            
    except TimeoutException:
        ret_score = timeout_score
    except Exception as e:
        error_msg = str(e)
        import traceback
        trace_msg = traceback.format_exc()
        try:
             with open("math_verify_error.log", "a") as f:
                 f.write(f"Error for GT: {ground_truth_boxed}, Pred: {processed_output}\n")
                 f.write(f"Exception: {error_msg}\n")
                 f.write(trace_msg + "\n")
        except:
             pass
        pass

    return {
        "score": ret_score,
        "canonical_pred": canonical_pred if canonical_pred is not None else processed_output,
        "canonical_gold": canonical_gold if canonical_gold is not None else ground_truth_boxed,
    }
