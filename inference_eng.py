import torch
from transformers import AutoTokenizer
from transformer import build_transformer
from utils import load_checkpoint
import argparse

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

def get_inference_parser():
    parser = argparse.ArgumentParser(description="Transformer Model Inference Script")
    parser.add_argument("--src_vocab_size", type=int, default=59514, help="Source vocabulary size")
    parser.add_argument("--tgt_vocab_size", type=int, default=59514, help="Target vocabulary size")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Dimensionality of the model")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of encoder and decoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimensionality of the feed-forward layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    return parser

def translate_sentence(model, src_sentence, max_len=512, pad_token=59513):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize input sentence
    inputs = tokenizer(src_sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    src = inputs["input_ids"].to(device)
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2).to(device)

    # Encode the source sentence
    encoder_output = model.encode(src, src_mask)

    # Prepare target input with <s> token
    tgt = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)

    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones((1, tgt.size(1), tgt.size(1)))).bool().to(device)
        output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
        logits = model.project(output)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode token IDs to text
    translated_tokens = tgt.squeeze(0).tolist()
    translated_text = tokenizer.decode(translated_tokens, skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    parser = get_inference_parser()
    args = parser.parse_args()

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
    print("Model loaded successfully. Ready for translation.")

    while True:
        src_sentence = input("Enter English sentence: ")
        if src_sentence.lower() == "exit":
            break
        translated_sentence = translate_sentence(model, src_sentence)
        print(f"Translated French sentence: {translated_sentence}")
