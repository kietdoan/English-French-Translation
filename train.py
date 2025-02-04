import torch
import torch.nn as nn
from transformer import build_transformer
from optimizer import create_optimizer_and_scheduler
import os
import argparse
from utils import create_causal_mask,create_padding_mask,load_checkpoint,save_checkpoint
from loader import get_trainloader,get_valloader
import tqdm
# import wandb
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def check_embeddings(model, vocab_size, device):
    """Check embeddings for meaningful values and detect potential issues."""
    with torch.no_grad():
        embedding_weights = model.src_embed.embedding.weight
        print("Embedding weight sample (first 5 rows):")
        print(embedding_weights[:5])

        # Check a sample input
        sample_input = torch.randint(0, vocab_size, (1, 5)).to(device)
        embedded_output = model.src_embed(sample_input)

        print("Embedded output shape:", embedded_output.shape)
        print("Sample embedding output:", embedded_output[0, :3])

        zero_count = (embedded_output == 0).sum().item()
        total_elements = embedded_output.numel()
        print(f"Zero values in embeddings: {zero_count}/{total_elements}")

        pad_token_embedding = model.src_embed.embedding.weight[59513]
        print("Padding token embedding:", pad_token_embedding)
        print("Padding token sum:", pad_token_embedding.sum().item())

        # Check gradients
        embedded_output.requires_grad_(True)
        loss = embedded_output.sum()
        loss.backward()
        print("Gradients in embedding layer (first 5 rows):")
        print(model.src_embed.embedding.weight.grad[:5])

def get_parser():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    
    # Dataset parameters
    parser.add_argument("--src_vocab_size", type=int, default=10000, help="Source vocabulary size")
    parser.add_argument("--tgt_vocab_size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")

    # Model parameters
    parser.add_argument("--d_model", type=int, default=512, help="Dimensionality of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder and decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimensionality of the feed-forward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Number of warmup steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")

    # File paths
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save training logs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint")
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train the model for one epoch with wandb and tqdm."""
    model.train()
    epoch_loss = 0

    progress_bar = tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        tgt_input = tgt[:, :-1]  # Teacher forcing input
        tgt_output = tgt[:, 1:]  # Expected output
        tgt_mask = tgt_mask[:, :, :-1, :-1]

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)
        # loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        # Update tqdm progress bar with loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # # Log every 10 batches to wandb
        # if batch_idx % 10 == 0:
        #     wandb.log({"train_loss": loss.item(), "batch": batch_idx + epoch * len(dataloader)})

    return epoch_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """Validate the model for one epoch with wandb and tqdm."""
    model.eval()
    epoch_loss = 0

    progress_bar = tqdm.tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=True)

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
            # Check gradients during training
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} gradient norm: {param.grad.norm().item()}")


            # Log every 10 batches to wandb
            # if batch_idx % 10 == 0:
            #     wandb.log({"val_loss": loss.item(), "batch": batch_idx + epoch * len(dataloader)})

    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading training data from: {args.train_dir}")
    train_dataloader = get_trainloader(args.train_dir,args.batch_size)
    val_dataloader = get_valloader(args.val_dir,args.batch_size)

    # Initialize the model
    print("Initializing the Transformer model...")
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
    
    # Call the function after model initialization
    # check_embeddings(model, args.src_vocab_size, device)
    # Create optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    # optimizer, scheduler = create_optimizer_and_scheduler(
    #     model,
    #     learning_rate=args.learning_rate,
    #     warmup_steps=args.warmup_steps,
    #     weight_decay=args.weight_decay,
    #     d_model=args.d_model
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=59513)  # Assuming 0 is the padding token
    # scheduler = get_warmup_scheduler(optimizer, warmup_steps=4000, total_steps=50000)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9, verbose=True)

    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            src_dummy = torch.randint(0, args.src_vocab_size, (1, args.max_length)).to(device)
            tgt_dummy = torch.randint(0, args.tgt_vocab_size, (1, args.max_length)).to(device)
            src_mask_dummy = torch.ones(1, 8, 1, args.max_length).to(device)
            tgt_mask_dummy = torch.ones(1, 8, args.max_length, args.max_length).to(device)

            with torch.no_grad():
                model(src_dummy, tgt_dummy, src_mask_dummy, tgt_mask_dummy)
                start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint) + 1
        else:
            print(f"Checkpoint {args.checkpoint} not found, starting from scratch.")

    # Training loop
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate for one epoch
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device, epoch)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Scheduler step
        scheduler.step()

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir)

    print("Training complete!")