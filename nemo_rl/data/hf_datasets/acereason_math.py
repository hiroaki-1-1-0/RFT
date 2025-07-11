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


from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_acereason_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    """Format AceReason-Math data into messages format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": str(data["answer"]),
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_acereason_math_dataset(
    seed: int = 42,
    test_size: float = 0.05,
) -> dict[str, Dataset | None]:
    """Load and split the AceReason-Math dataset into train and validation sets using HF's train_test_split."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset (only has train split)
    original_ds = load_dataset("nvidia/AceReason-Math", split="train")

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(
        format_acereason_math,
        remove_columns=split_ds["train"].column_names,
    )
    val_formatted = split_ds["test"].map(
        format_acereason_math,
        remove_columns=split_ds["test"].column_names,
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class AceReasonMathDataset:
    def __init__(
        self,
        seed: int = 42,
        test_size: float = 0.05,
        prompt_file: Optional[str] = None,
    ):
        """Initialize the AceReason-Math dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
            prompt_file: Optional path to prompt file
        """
        self.formatted_ds = prepare_acereason_math_dataset(
            seed=seed, test_size=test_size
        )

        self.task_spec = TaskDataSpec(
            task_name="AceReason-Math",
            prompt_file=prompt_file,
        ) 