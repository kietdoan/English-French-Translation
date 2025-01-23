import torch
import json
from torch.utils.data import Dataset, DataLoader
from utils import create_causal_mask, create_padding_mask

class TranslationDataset(Dataset):
    """Custom dataset for translation tasks with JSON format."""
    def __init__(self, json_file, pad_token=59513):
        """
        Initialize the dataset.
        Args:
            json_file (str): Path to the JSON file containing translation pairs.
            pad_token (int): Token used for padding.
        """
        self.pad_token = pad_token
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f)  # Load entire JSON array
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {e}")
                raise

        # Check if the JSON data contains expected keys
        for entry in self.data:
            if "input_ids" not in entry or "labels" not in entry:
                raise ValueError("Each JSON entry must contain 'input_ids' and 'labels' keys.")

    def __len__(self):
        """Return the total number of sentence pairs."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sentence pair from the JSON data.
        Args:
            idx (int): Index of the pair to retrieve.
        Returns:
            tuple: A tuple containing (source_sentence, target_sentence).
        """
        src = self.data[idx]["input_ids"]  # Tokenized source sentence
        tgt = self.data[idx]["labels"]     # Tokenized target sentence

        # Add padding token to the start of target sequences
        tgt = [self.pad_token] + tgt

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch, pad_token=59513):
    """Collate function to pad sequences and create masks."""
    src_sentences, tgt_sentences = zip(*batch)

    # Find max length of src and tgt sequences
    src_max_len = max(len(src) for src in src_sentences)
    tgt_max_len = max(len(tgt) for tgt in tgt_sentences)

    # Create padded tensors
    src_padded = torch.full((len(batch), src_max_len), pad_token, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_token, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_sentences, tgt_sentences)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    # Create masks
    src_mask = create_padding_mask(src_padded, pad_token=pad_token)
    tgt_padding_mask = create_padding_mask(tgt_padded, pad_token=pad_token)
    tgt_causal_mask = create_causal_mask(tgt_max_len)

    # Combine padding and causal masks for tgt
    tgt_mask = tgt_padding_mask & tgt_causal_mask

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,  # Encoder padding mask
        "tgt_mask": tgt_mask,  # Decoder mask (padding + causal)
    }

def get_trainloader(json_file, batch_size):
    """
    Create a DataLoader for the training dataset.
    """
    dataset = TranslationDataset(json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def get_valloader(json_file, batch_size):
    """
    Create a DataLoader for the validation dataset.
    """
    dataset = TranslationDataset(json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
