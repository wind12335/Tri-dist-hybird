################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

class AutoLLM:
    model_mapping = {
        "Qwen/Qwen3-0.6B": "dense",
        "Qwen/Qwen3-8B": "dense",
        "Qwen/Qwen3-14B": "dense",
        "Qwen/Qwen3-32B": "dense",
        "Qwen/Qwen3-30B-A3B": "qwen_moe",
        "Qwen/Qwen3-235B-A22B": "qwen_moe",
        "meta-llama/Meta-Llama-3-70B": "dense",
        "ByteDance-Seed/Seed-OSS-36B-Instruct": "dense",
    }

    @staticmethod
    def _load_model_class(model_kind: str):
        if model_kind == "dense":
            from .dense import DenseLLM
            return DenseLLM
        if model_kind == "qwen_moe":
            from .qwen_moe import Qwen3MoE
            return Qwen3MoE
        raise ValueError(f"Unsupported model kind: {model_kind}")

    @staticmethod
    def from_pretrained(config, group=None):
        model_name = config.model_name

        for standard_name, model_kind in AutoLLM.model_mapping.items():
            if model_name.endswith(standard_name):
                model_cls = AutoLLM._load_model_class(model_kind)
                return model_cls(config, group)

        if model_name in AutoLLM.model_mapping:
            model_cls = AutoLLM._load_model_class(AutoLLM.model_mapping[model_name])
            return model_cls(config, group)
        else:
            model_cls = AutoLLM._load_model_class("dense")
            print(f"Model {model_name} not found in model mapping, "
                  f"Available models: {list(AutoLLM.model_mapping.keys())} "
                  f"Falling back to DenseLLM with default configuration.")
            return model_cls(config, group)


class AutoTokenizer:

    def __init__(self):
        self.tokenizer = None

    @staticmethod
    def from_pretrained(model_config):
        from transformers import AutoTokenizer as HFTokenizer

        return HFTokenizer.from_pretrained(model_config.model_name, use_fast=True, legacy=False,
                                           local_files_only=model_config.local_only)
