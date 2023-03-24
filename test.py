import unittest
import dataset
from dataset import load_dataset
import torch


class TestDatasetIdxSplit(unittest.TestCase):
    def runTest(self):
        d = dataset.Dataset("HELLO WORLD")
        d.label = torch.arange(25)
        train_indices, valid_indices, test_indices = d.get_idx_split(0.5, 0.25)
        self.assertEqual(train_indices.shape[0], 12)

class TestDatasetGenerateArxiv(unittest.TestCase):
    def runTest(self):
        dataset = load_dataset("ogbn-arxiv")
        train_idx, valid_idx, test_idx =  dataset.get_idx_split()
        print(dataset.graph['edge_index'])
        
        for key, value in dataset.graph.items():
            print(key, value)
            print("_________")

        self.assertEqual(train_idx.shape[0], 90941)

if __name__ == "__main__":
    unittest.main()



