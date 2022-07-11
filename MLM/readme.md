# Training CUI BERT

- run train_tokenizer.py which traines a tokenizer on text
- run cui_tokenizer.py which replaces the vocabulary with CUIs ('C' stripped)
- copy cui_tokenizer.json to Tokenizer/tokenizer.json
- run make_samples.py to make training data (notes.txt)
- replace tokenizer_name in run.sh with Tokenizer
- set batch size to 28 to train on 6 GPUs
