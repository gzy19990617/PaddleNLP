# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from paddlenlp.experimental.transformers.deepseek_v2.modeling import (
    DeepseekV2ForCausalLMBlockInferenceModel,
)
from paddlenlp.transformers import DeepseekV2Config

__all__ = ["DeepseekV3ForCausalLMBlockInferenceModel"]


class DeepseekV3ForCausalLMBlockInferenceModel(DeepseekV2ForCausalLMBlockInferenceModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.deepseek_v2.base_model_prefix = "deepseek_v3"
