import unittest
from itertools import islice
from bucket_batch_sampler import BucketBatchSampler, _FakeDataset


class TestBucketBatchSampler(unittest.TestCase):

    def setUp(self):
        # 10 items, two buckets: 'A' (7 items) and 'B' (3 items)
        rows = [{'bucket': 'A'}] * 7 + [{'bucket': 'B'}] * 3
        self.dataset = _FakeDataset(rows)

    # -----------------------------------------------------------------
    # utility
    # -----------------------------------------------------------------
    def _collect_batches(self, sampler):
        return list(sampler)

    # -----------------------------------------------------------------
    # Original test cases
    # -----------------------------------------------------------------
    def test_batch_size_4_first_epoch(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=4)
        batches = self._collect_batches(sampler)
        lengths = [len(b) for b in batches]
        # we expect 3 batches: 4, 4, 2 (10 items total)
        self.assertEqual(sum(lengths), 10)
        self.assertEqual(max(lengths), 4)
        self.assertLessEqual(min(lengths), 4)

    def test_batch_size_3_with_leftovers(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=3)
        batches = self._collect_batches(sampler)
        lengths = [len(b) for b in batches]
        self.assertEqual(sum(lengths), 10)
        self.assertEqual(max(lengths), 3)

    def test_leftovers_are_reused_next_epoch(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=6)
        # first epoch: 6 + 4
        batches1 = self._collect_batches(sampler)
        # second epoch: also 10 items. The implementation does not carry over leftovers.
        batches2 = self._collect_batches(sampler)
        self.assertEqual(sum(map(len, batches1)), 10)
        self.assertEqual(sum(map(len, batches2)), 10)

    def test_all_samples_emitted(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=12)
        # one epoch only
        batches = self._collect_batches(sampler)
        self.assertEqual(sum(map(len, batches)), 10)

    def test_empty_leftover_list_after_iteration(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=4)
        _ = list(sampler)
        self.assertEqual(sampler.leftover_items, [])

    def test_len_estimate(self):
        sampler = BucketBatchSampler(self.dataset, batch_size=4)
        self.assertEqual(len(sampler), 3)  # ceil(10/4)

    # -----------------------------------------------------------------
    # New test cases
    # -----------------------------------------------------------------
    def test_invalid_batch_size(self):
        """Tests that a non-positive batch size raises a ValueError."""
        with self.assertRaises(ValueError):
            BucketBatchSampler(self.dataset, batch_size=0)
        with self.assertRaises(ValueError):
            BucketBatchSampler(self.dataset, batch_size=-5)

    def test_empty_dataset(self):
        """Tests that the sampler handles an empty dataset gracefully."""
        empty_dataset = _FakeDataset([])
        sampler = BucketBatchSampler(empty_dataset, batch_size=4)
        batches = self._collect_batches(sampler)
        self.assertEqual(len(sampler), 0)
        self.assertEqual(len(batches), 0)

    def test_exact_multiple_batch_size(self):
        """Tests when the dataset size is a perfect multiple of the batch size."""
        rows = [{'bucket': 'A'}] * 8 + [{'bucket': 'B'}] * 4 # 12 items
        dataset = _FakeDataset(rows)
        sampler = BucketBatchSampler(dataset, batch_size=4)
        batches = self._collect_batches(sampler)
        lengths = [len(b) for b in batches]

        self.assertEqual(len(sampler), 3) # 12 / 4 = 3
        self.assertEqual(sum(lengths), 12)
        self.assertTrue(all(l == 4 for l in lengths))

    def test_single_bucket(self):
        """Tests when all data belongs to a single bucket."""
        rows = [{'bucket': 'A'}] * 15
        dataset = _FakeDataset(rows)
        sampler = BucketBatchSampler(dataset, batch_size=4)
        batches = self._collect_batches(sampler)
        lengths = [len(b) for b in batches]

        self.assertEqual(sum(lengths), 15)
        self.assertEqual(len(batches), 4) # 4, 4, 4, 3

    def test_randomness_across_epochs(self):
        """Tests that batches are shuffled differently in consecutive epochs."""
        sampler = BucketBatchSampler(self.dataset, batch_size=2)
        batches1 = self._collect_batches(sampler)
        batches2 = self._collect_batches(sampler)

        # It's statistically very improbable for these to be identical
        # if shuffling is working correctly.
        self.assertNotEqual(batches1, batches2)


if __name__ == '__main__':
    unittest.main()