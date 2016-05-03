# Recurrent Memory Networks

This code implements Recurrent Memory Networks (RM and RMR) described in

	Recurrent Memory Networks for Language Modeling 
	Ke Tran, Arianna Bisazza, and Christof Monz 
	In Proceedings of NAACL 2016


Much of this code is based on [char-nn](https://github.com/karpathy/char-rnn).


## Use the code

Here is an example how to run the code. 

```
$ th RM.lua -max_seq_length 80 -min_seq_length 10 -max_epochs 20 \
-data_dir data/it -print_every 200 -num_layers 1 -mem_size 15  \
-learning_rate 1 -emb_size 128 -rnn_size 128  -nhop 1 \
-checkpoint_dir checkpoint
```

There are many settings we didn't perform in our paper, for example increasing the embedding size, stacking multiple LSTM layers, etc. The reason is that we have a limitted time to spend on shared GPUs.

## Data Processing
Since we are interested mostly in long distant dependencies, we only selected sentences whose length is in between 10 and 80 tokens (use this option `-max_seq_length 80 -min_seq_length 10`). If you want to compare to our results, ideally, you should use the same setting.


## TODO:
- clean up the code (remove some experimental settings)
- fast output layer?