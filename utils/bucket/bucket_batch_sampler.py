import random
import math
from collections import defaultdict
from typing import Dict, List, Iterator, Any, Tuple

from typing import Iterator
from torch.utils.data import Sampler

class _FakeDataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.datarows = rows

    def __len__(self):
        return len(self.datarows)


class BucketBatchSampler:
    def __init__(self, dataset, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.datarows = dataset.datarows
        self.batch_size = batch_size
        self.leftover_items: Dict[str, List[int]] = defaultdict(list)

    # -------------------------------------------------------------
    def _bucket_indices(self) -> Dict[str, List[int]]:
        buckets = defaultdict(list)
        for idx, row in enumerate(self.datarows):
            # print("row:", row)
            key = row['bucket']
            buckets[key].append(idx)
        for indices in buckets.values():
            random.shuffle(indices)
        return dict(buckets)

    def __len__(self) -> int:
        return math.ceil(len(self.datarows) / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        # 1. Start with fresh buckets
        buckets = defaultdict(list, self._bucket_indices())

        # 2. Re-insert leftovers into their respective buckets
        for key, indices in self.leftover_items.items():
            buckets[key][:0] = indices  # prepend
        self.leftover_items.clear()

        # 3. Shuffle bucket order
        bucket_list = list(buckets.items())
        random.shuffle(bucket_list)

        # 4. Yield full batches per bucket
        for key, indices in bucket_list:
            indices = list(indices)  # copy
            start = 0
            while start + self.batch_size <= len(indices):
                yield indices[start:start + self.batch_size]
                start += self.batch_size
            if start < len(indices):
                self.leftover_items[key].extend(indices[start:])

        # 5. Flush leftovers bucket-by-bucket
        for key, indices in self.leftover_items.items():
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]
        self.leftover_items.clear()
        

class FlatBatchSampler:  # Renamed to reflect flat sampling; change back if needed
    def __init__(self, dataset, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.datarows = dataset.datarows
        self.batch_size = batch_size
        self.indices = list(range(len(self.datarows)))

    def __len__(self) -> int:
        return math.ceil(len(self.datarows) / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle all indices globally each epoch
        shuffled = self.indices.copy()
        random.shuffle(shuffled)

        # Yield batches sequentially
        for i in range(0, len(shuffled), self.batch_size):
            yield shuffled[i:i + self.batch_size]