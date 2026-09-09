from tokenizers import Tokenizer
import tiktoken
import torch

tiktokn = tiktoken.get_encoding("gpt2")
tokenizer = Tokenizer.from_file("Tokenizer/tokenizer.json")

text = "Привет, мир"

encoded = tokenizer.encode(text)

print("Encoded obj: ", encoded)
print("Encoded tokens: ", encoded.tokens)
print("Encoded ids: ", encoded.ids)
enc_tensor = torch.tensor(encoded.ids).unsqueeze(0)
print("Torch Unsqz", enc_tensor)

flat = enc_tensor.squeeze(0)
decoded = tokenizer.decode(flat.tolist())

print("Decoded: ", decoded)

print("\n\n\n", "==" * 120, "\n\n\n")

encoded = tiktokn.encode(text)

print("Encoded obj: ", encoded)
print(torch.tensor(encoded).unsqueeze(0))

decoded = tiktokn.decode(encoded)

print("Decoded: ", decoded)