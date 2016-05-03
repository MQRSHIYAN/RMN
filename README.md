# Recurrent Memory Network

This code implement the Recurrent Memory Network described in

	Recurrent Memory Network for Language Modeling. Ke Tran, Arianna Bisazza, and Christof Monz. InProceedings of the North American Chapter of the Association for Computational Linguistics (NAACL-2016), to appear.


Much of this code is based on [char-nn](https://github.com/karpathy/char-rnn).


## Use the code

Here is an example how to run the code. 

```
$ th RM.lua -max_seq_length 80 -min_seq_length 10 -max_epochs 20 -data_dir data/it -print_every 200 -num_layers 1 -mem_size 15  -learning_rate 1 -emb_size 128 -rnn_size 128  -nhop 1 -checkpoint_dir checkpoint
```

There are many settings we didn't perform in our paper, for example increasing the embedding size, stacking multiple LSTM layers, etc. The reason is that we have a limitted time to spend on shared GPUs.

## Data Processing
Since we are interested mostly in long distant dependencies, we only select sentences whose lengths are in between 10 and 80 tokens. If you want to compare to our results, ideally, you should use the same setting.


## TODO:
- clean up the code (remove some experimental setting)
- fast output layer?