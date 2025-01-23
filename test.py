import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import os
from transformer import build_transformer
from utils import load_checkpoint
from loader import get_valloader
from torch.nn.functional import cross_entropy
import argparse

def get_test_parser():
    parser = argparse.ArgumentParser(description="Transformer Model Testing Script")

    # Model parameters
    parser.add_argument("--src_vocab_size", type=int, default=59514, help="Source vocabulary size")
    parser.add_argument("--tgt_vocab_size", type=int, default=59514, help="Target vocabulary size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Dimensionality of the model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder and decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimensionality of the feed-forward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    # File paths
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint")

    return parser

def test_model(model, dataloader, criterion, device):
    """Validate the model for one epoch with wandb and tqdm."""
    model.eval()
    epoch_loss = 0

    progress_bar = tqdm.tqdm(dataloader, desc=f"Test model", leave=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask = tgt_mask[:, :, :-1, :-1]

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            epoch_loss += loss.item()

            # Update tqdm progress bar with loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(dataloader)

def calculate_bleu_score(predictions, references):
    """Calculate BLEU score using nltk."""
    from nltk.translate.bleu_score import sentence_bleu
    bleu_scores = [sentence_bleu([ref], pred) for pred, ref in zip(predictions, references)]
    return sum(bleu_scores) / len(bleu_scores)

if __name__ == "__main__":
    # Parse arguments
    parser = get_test_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading test data from: {args.test_dir}")
    test_dataloader = get_valloader(args.test_dir, batch_size=1)
    print("Loading trained model...")
    model = build_transformer(
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        src_seq_len=args.max_length,
        tgt_seq_len=args.max_length,
        d_model=args.d_model,
        N=args.num_layers,
        head=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout
    ).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Running one batch of random data to initialize model parameters...")
    src_dummy = torch.randint(0, args.src_vocab_size, (1, args.max_length)).to(device)
    tgt_dummy = torch.randint(0, args.tgt_vocab_size, (1, args.max_length)).to(device)
    src_mask_dummy = torch.ones(1, 8, 1, args.max_length).to(device)
    tgt_mask_dummy = torch.ones(1, 8, args.max_length, args.max_length).to(device)

    with torch.no_grad():
        model(src_dummy, tgt_dummy, src_mask_dummy, tgt_mask_dummy)
    start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)
    print(f"Model loaded from checkpoint. Resuming from epoch {start_epoch}")
    criterion = nn.CrossEntropyLoss(ignore_index=59513)
    #Test the model
    test_loss = test_model(model, test_dataloader,criterion, device)
    print(f"Test Loss: {test_loss:.4f}")