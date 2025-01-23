import torch
import argparse
from transformer import build_transformer
from utils import load_checkpoint, create_padding_mask, create_causal_mask

def get_inference_parser():
    parser = argparse.ArgumentParser(description="Transformer Model Inference Script")

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
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint")

    return parser

def translate_token_ids(model, token_ids, device, max_len=512, pad_token=59513, end_token=0):
    model.eval()

    # Pad input token IDs if shorter than max length
    token_ids = token_ids + [pad_token] * (max_len - len(token_ids))
    src_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Create source mask
    src_mask = create_padding_mask(src_tensor, pad_token=pad_token)

    # Encode the source sequence
    encoder_output = model.encode(src_tensor, src_mask)

    # Start decoding with <START> token
    tgt_tokens = [pad_token]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_pad_mask = create_padding_mask(tgt_tensor, pad_token=pad_token)
        tgt_causal_mask = create_causal_mask(tgt_tensor.size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask

        # Decode
        decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, tgt_mask)
        logits = model.project(decoder_output)

        # Get the predicted token (last position)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tgt_tokens.append(next_token)

        # Stop when the <END> token is predicted
        if next_token == end_token:
            break

    return tgt_tokens

if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    src_dummy = torch.randint(0, args.src_vocab_size, (1, args.max_length)).to(device)
    tgt_dummy = torch.randint(0, args.tgt_vocab_size, (1, args.max_length)).to(device)
    src_mask_dummy = torch.ones(1, 8, 1, args.max_length).to(device)
    tgt_mask_dummy = torch.ones(1, 8, args.max_length, args.max_length).to(device)

    with torch.no_grad():
        model(src_dummy, tgt_dummy, src_mask_dummy, tgt_mask_dummy)
    start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)
    print(f"Model loaded from checkpoint. Resuming from epoch {start_epoch}")

    # Get token IDs input from user
    input_tokens = input("Enter a sequence of token IDs (comma-separated): ")
    token_ids = list(map(int, input_tokens.strip().split(',')))

    translated_tokens = translate_token_ids(model, token_ids, device)
    print(f"Translated Token IDs: {translated_tokens}")
