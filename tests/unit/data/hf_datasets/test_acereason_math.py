# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from transformers import AutoTokenizer

from nemo_rl.data.hf_datasets.acereason_math import AceReasonMathDataset


@pytest.mark.skip(reason="dataset download is flaky")
def test_acereason_math_dataset():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    acereason_dataset = AceReasonMathDataset()

    # Check that the dataset is formatted correctly
    for example in acereason_dataset.formatted_ds["train"].take(5):
        assert "messages" in example
        assert "task_name" in example
        assert example["task_name"] == "math"
        assert len(example["messages"]) == 2

        assert example["messages"][0]["role"] == "user"
        assert example["messages"][1]["role"] == "assistant"

        # Check that content is not empty
        assert len(example["messages"][0]["content"]) > 0
        assert len(example["messages"][1]["content"]) > 0

    # Check that validation dataset exists
    assert acereason_dataset.formatted_ds["validation"] is not None
    assert len(acereason_dataset.formatted_ds["validation"]) > 0

    # Check task spec
    assert acereason_dataset.task_spec.task_name == "AceReason-Math"


@pytest.mark.skip(reason="dataset download is flaky")
def test_acereason_math_dataset_with_custom_split():
    # Test with custom test size
    acereason_dataset = AceReasonMathDataset(test_size=0.1, seed=123)

    train_size = len(acereason_dataset.formatted_ds["train"])
    val_size = len(acereason_dataset.formatted_ds["validation"])
    
    # Check that validation is approximately 10% of total
    total_size = train_size + val_size
    val_ratio = val_size / total_size
    assert 0.08 < val_ratio < 0.12  # Allow some tolerance due to rounding 