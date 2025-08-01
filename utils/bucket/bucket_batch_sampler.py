import random
import math
from collections import defaultdict
from typing import Dict, List, Iterator, Any

class BucketBatchSampler:
    def __init__(self, dataset, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.datarows = dataset.datarows
        self.batch_size = batch_size
        self.leftover_items: List[int] = []
        self.bucket_indices: Dict[str, List[int]] = self._bucket_indices()

    # -------------------------------------------------------------
    def _bucket_indices(self) -> Dict[str, List[int]]:
        buckets = defaultdict(list)
        for idx, row in enumerate(self.datarows):
            key = row['bucket']
            buckets[key].append(idx)
        for indices in buckets.values():
            random.shuffle(indices)
        return dict(buckets)

    def __len__(self) -> int:
        """
        Estimates the number of batches in one epoch.
        """
        return math.ceil(len(self.datarows) / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        # 1. Start with fresh buckets from the dataset
        buckets = self._bucket_indices()

        # 2. Prepend leftover items from the previous epoch
        for idx in self.leftover_items:
            key = self.datarows[idx]['bucket']
            buckets.setdefault(key, []).insert(0, idx)
        self.leftover_items.clear()

        # 3. Shuffle bucket order
        bucket_list = list(buckets.items())
        random.shuffle(bucket_list)

        # 4. Yield batches from each bucket
        for _, indices in bucket_list:
            indices = list(indices)  # Work on a copy
            start = 0
            while start < len(indices):
                end = start + self.batch_size
                batch = indices[start:end]
                if end <= len(indices):  # Full batch
                    yield batch
                else:  # Leftover items
                    self.leftover_items.extend(batch)
                start = end

        # 5. Flush any remaining leftovers in correct-sized chunks
        while self.leftover_items:
            batch = self.leftover_items[:self.batch_size]
            yield batch
            self.leftover_items = self.leftover_items[self.batch_size:]