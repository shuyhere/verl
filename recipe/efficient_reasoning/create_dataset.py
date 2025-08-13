#!/usr/bin/env python3
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
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

import argparse
import os
import pandas as pd
from typing import List, Dict, Any


def convert_rows_from_hfds(ds, split: str) -> List[Dict[str, Any]]:
    """Convert HF saved dataset to Verl RLHF chat schema expected by RLHFDataset.

    Output dict per row:
      - data_source: 'datasets/compression_dataset'
      - prompt: list of chat messages [{'role':'user','content': ...}]
      - ability: 'math'
      - reward_model: {'style':'rule','ground_truth':[extracted]}
      - extra_info: {'question': problem, 'split': split}
    """
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        problem = ex.get('problem', '')
        extracted = ex.get('extracted', None)
        gt = None if extracted is None else str(extracted)
        user_prompt = (
            "Please reason step by step, and put your final answer within \\boxed{{}}. "
            f"Question: {problem}"
        )
        rows.append(
            {
                "data_source": "datasets/compression_dataset",
                "prompt": [{"role": "user", "content": user_prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": [gt]},
                "extra_info": {"question": problem, "split": split},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to HF save_to_disk dataset dir')
    parser.add_argument('--output', required=True, help='Output Parquet path for train or full dataset')
    parser.add_argument('--split', default='train', help='Split name to annotate (default: train)')
    parser.add_argument('--val_output', default=None, help='Optional: Output Parquet path for validation split')
    parser.add_argument('--val_ratio', type=float, default=None, help='Optional: Validation ratio, e.g., 0.1 for 9:1 split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    args = parser.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.input)
    if 'train' in ds:
        ds = ds['train']

    # If val_ratio and val_output provided, perform split
    if args.val_output and args.val_ratio is not None and 0.0 < args.val_ratio < 1.0:
        split_ds = ds.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_ds = split_ds['train']
        val_ds = split_ds['test']

        train_rows = convert_rows_from_hfds(train_ds, split='train')
        val_rows = convert_rows_from_hfds(val_ds, split='validation')

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        os.makedirs(os.path.dirname(args.val_output), exist_ok=True)

        from datasets import Dataset
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)
        Dataset.from_pandas(train_df, preserve_index=False).to_parquet(args.output)
        Dataset.from_pandas(val_df, preserve_index=False).to_parquet(args.val_output)
        print(f"Wrote {len(train_df)} train rows to {args.output}")
        print(f"Wrote {len(val_df)} val rows to {args.val_output}")
    else:
        rows = convert_rows_from_hfds(ds, split=args.split)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        # Write Parquet using nested structures compatible with Hugging Face Datasets
        from datasets import Dataset
        df = pd.DataFrame(rows)
        Dataset.from_pandas(df, preserve_index=False).to_parquet(args.output)
        print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == '__main__':
    main()


