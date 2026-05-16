# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Stopping criteria for rollout generation to detect and abort problematic outputs.

This module provides detection mechanisms for:
1. Multiple boxed answers (model outputs boxed answer but continues generating)
2. Excessive tokens after first boxed answer
3. Re-thinking patterns after answer completion
4. Unreasonably long boxed answers
"""

import re
from typing import Optional


class RepetitionDetector:
    """Detect problematic patterns where model continues generating after completing the answer."""

    def __init__(
        self,
        max_tokens_after_box: int = 200,
        detect_multiple_boxes: bool = True,
        detect_rethink_patterns: bool = True,
        max_box_length: int = 500,
    ):
        """Initialize the repetition detector.

        Args:
            max_tokens_after_box: Maximum tokens allowed after first boxed answer (估算值)
            detect_multiple_boxes: Whether to abort on multiple boxed answers
            detect_rethink_patterns: Whether to detect re-thinking patterns after answer
            max_box_length: Maximum reasonable length for a single answer in tokens (估算值)
        """
        self.max_tokens_after_box = max_tokens_after_box
        self.detect_multiple_boxes = detect_multiple_boxes
        self.detect_rethink_patterns = detect_rethink_patterns
        self.max_box_length = max_box_length
        
        # Patterns that indicate model is re-thinking after already giving answer
        self.rethink_patterns = [
            r"let me think again",
            r"let'?s recalculate",
            r"wait,?\s*(?:let me|i should)",
            r"actually,?\s*(?:let me|i need to)",
            r"hmm+,?\s*(?:let me|maybe)",
            r"(?:let me|let'?s)\s+(?:re-?)?(?:do|check|verify|calculate)",
            r"i made (?:a |an )?(?:mistake|error)",
            r"(?:let me|let'?s)\s+(?:try|start)\s+(?:again|over)",
        ]

    def check_repetition(self, response_text: str, tokenizer=None, token_ids=None) -> tuple[bool, Optional[str]]:
        """
        检查是否应该中止生成。
        
        Args:
            response_text: 生成的文本
            tokenizer: tokenizer 实例，用于准确计算 token 数量
            token_ids: 已生成的 token IDs
        
        主要检测模型输出答案后继续废话的情况：
        - 已经输出了\boxed{}后继续生成过多token
        - 已经输出了\boxed{}后出现"let me think again"等重新思考的pattern
        - 单个\boxed{}答案过长
        
        注意：不再检测多个\boxed{}，因为Math-Verify可以正确处理（取最后一个）
        
        Returns:
            (should_abort: bool, reason: Optional[str])
        """
        # 检查1：移除多个boxed检测 - Math-Verify现在可以正确处理多个\boxed{}（取最后一个）
        # if self.detect_multiple_boxes and self._check_multiple_boxes(response_text, token_ids):
        #     return True, "multiple_boxed_answers"
        
        # 检查2：第一个boxed答案后是否生成了过多token
        if self._check_excessive_tokens_after_box(response_text, tokenizer, token_ids):
            return True, "excessive_tokens_after_answer"
        
        # 检查3：答案后是否出现重新思考的pattern
        if self.detect_rethink_patterns and self._check_rethink_after_box(response_text, token_ids):
            return True, "rethinking_after_answer"
        
        # 检查4：单个boxed答案是否过长
        if self._check_box_too_long(response_text):
            return True, "boxed_answer_too_long"
        
        return False, None

    def _find_first_box_position(self, text: str, token_ids: list = None) -> Optional[int]:
        """
        找到第一个完整的box答案的结束位置。
        支持两种格式：
        1. Special token: <|box_start|>...<|box_end|> (Qwen3)
        2. LaTeX: \boxed{...} 或 \box{...}
        
        Args:
            text: 生成的文本
            token_ids: token序列，用于检测special token
            
        Returns:
            第一个box结束的字符位置，如果没有则返回None
        """
        # 方案1: Token层面检测 <|box_end|> (token ID: 151649)
        # 这是最可靠的方法，不受文本格式影响
        if token_ids is not None:
            BOX_END_TOKEN = 151649  # Qwen3的<|box_end|> token
            if BOX_END_TOKEN in token_ids:
                # 找到第一个box_end的位置
                # 注意：我们需要返回文本位置，所以要解码到该位置
                return -1  # 标记：使用token检测成功，返回特殊值
        
        # 方案2: 文本层面正则检测 \boxed{}
        # Fallback方案，兼容非special token输出
        pattern = r'\\box(?:ed)?\{[^{}]*\}'
        match = re.search(pattern, text)
        if match:
            return match.end()  # 返回第一个box结束的位置
        
        return None

    def _count_boxes(self, text: str, token_ids: list = None) -> int:
        """
        统计完整的box答案数量。
        优先使用token层面检测。
        """
        # 方案1: Token层面检测
        if token_ids is not None:
            BOX_END_TOKEN = 151649  # Qwen3的<|box_end|>
            count = token_ids.count(BOX_END_TOKEN)
            if count > 0:
                return count
        
        # 方案2: 文本正则检测
        pattern = r'\\box(?:ed)?\{[^{}]*\}'
        return len(re.findall(pattern, text))

    def _check_multiple_boxes(self, text: str, token_ids: list = None) -> bool:
        """检查是否有多个boxed答案（说明模型在重复输出答案）"""
        box_count = self._count_boxes(text, token_ids)
        return box_count > 1

    def _check_excessive_tokens_after_box(self, text: str, tokenizer=None, token_ids=None) -> bool:
        """
        检查第一个boxed答案后是否生成了过多内容。
        优先使用token层面检测，精确且不受文本格式影响。
        """
        # 方案1: Token层面检测（最可靠）
        if token_ids is not None:
            BOX_START_TOKEN = 151648  # Qwen3的<|box_start|> 
            BOX_END_TOKEN = 151649    # Qwen3的<|box_end|>
            
            # 找到第一个box_end的位置
            if BOX_END_TOKEN in token_ids:
                box_end_idx = token_ids.index(BOX_END_TOKEN)
                tokens_after_box = len(token_ids) - box_end_idx - 1
                return tokens_after_box > self.max_tokens_after_box
        
        # 方案2: 文本层面检测（fallback）
        first_box_end = self._find_first_box_position(text, token_ids)
        if first_box_end is None:
            return False  # 还没有boxed答案，不检查
        
        if first_box_end == -1:
            # 已经在token层面检测过了
            return False
        
        # 使用真实 tokenizer 计算 token 数量
        if tokenizer is not None:
            after_box = text[first_box_end:]
            after_box_token_ids = tokenizer.encode(after_box, add_special_tokens=False)
            actual_tokens = len(after_box_token_ids)
            return actual_tokens > self.max_tokens_after_box
        else:
            # Fallback: 粗略估算（如果没有 tokenizer）
            after_box = text[first_box_end:]
            estimated_tokens = len(after_box.split()) * 1.3
            return estimated_tokens > self.max_tokens_after_box

    def _check_rethink_after_box(self, text: str, token_ids: list = None) -> bool:
        """检查第一个boxed答案后是否出现重新思考的pattern"""
        first_box_end = self._find_first_box_position(text, token_ids)
        if first_box_end is None:
            return False  # 还没有boxed答案，不检查
        
        if first_box_end == -1:
            # Token检测到了box，但我们需要文本来检查rethink pattern
            # 这种情况下，检查整个文本中是否有box token之后的rethink pattern
            if token_ids is not None:
                BOX_END_TOKEN = 151649
                if BOX_END_TOKEN in token_ids:
                    # 简化：检查整个文本，因为我们知道有box
                    text_lower = text.lower()
                    for pattern in self.rethink_patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            return True
            return False
        
        # 获取第一个box之后的文本
        after_box = text[first_box_end:].lower()
        
        # 检查是否匹配任何rethink pattern
        for pattern in self.rethink_patterns:
            if re.search(pattern, after_box, re.IGNORECASE):
                return True
        
        return False

    def _check_box_too_long(self, text: str) -> bool:
        """检查单个boxed答案是否过长（可能是模型在box内重复废话）"""
        pattern = r'\\box(?:ed)?\{([^{}]*)\}'
        boxes = re.findall(pattern, text)
        
        for box_content in boxes:
            # 粗略估算token数
            estimated_tokens = len(box_content.split()) * 1.3
            if estimated_tokens > self.max_box_length:
                return True
        
        return False


def should_abort_generation(
    response_text: str,
    tokens_generated: int,
    max_tokens: int = 4096,
    max_tokens_after_box: int = 200,
    detect_multiple_boxes: bool = True,
    detect_rethink_patterns: bool = True,
    max_box_length: int = 500,
    tokenizer=None,
    token_ids=None,
    **kwargs
) -> tuple[bool, Optional[str]]:
    """
    主函数：判断是否应该中止生成。
    
    Args:
        response_text: 已生成的文本
        tokens_generated: 已生成的token数
        max_tokens: 最大允许的token数
        max_tokens_after_box: 第一个boxed答案后最多允许的token数
        detect_multiple_boxes: 是否检测多个boxed答案
        detect_rethink_patterns: 是否检测重新思考的pattern
        max_box_length: 单个boxed答案最大长度
        tokenizer: tokenizer 实例，用于准确计算 token 数量
        token_ids: 已生成的 token IDs
        **kwargs: 其他参数
    
    Returns:
        (should_abort: bool, reason: Optional[str])
    """
    # 检查1：是否超过最大长度
    if tokens_generated >= max_tokens:
        return True, "max_length_exceeded"
    
    # 检查2：使用RepetitionDetector检测各种问题pattern
    detector = RepetitionDetector(
        max_tokens_after_box=max_tokens_after_box,
        detect_multiple_boxes=detect_multiple_boxes,
        detect_rethink_patterns=detect_rethink_patterns,
        max_box_length=max_box_length,
    )
    
    should_abort, reason = detector.check_repetition(response_text, tokenizer, token_ids)
    if should_abort:
        return True, reason
    
    return False, None
