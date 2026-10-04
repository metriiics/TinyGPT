from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers as pre_tok
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel


tokenizer = Tokenizer(
    BPE(unk_token="<unk>")
    )

trainer = BpeTrainer(
    vocab_size=21823, 
    special_tokens=["<unk>", "<pad>", "<eos>"],
    min_frequency=2
)

tokenizer.pre_tokenizer = pre_tok.ByteLevel()
tokenizer.decoder = ByteLevel()

tokenizer.train(["output.txt"], trainer)

tokenizer.save("Tokenizer/vocabulary/tokenizer.json")