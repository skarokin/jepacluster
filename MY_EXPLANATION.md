# high-level goal

at a high level, we learn how to cluster logs by future state by:
- read a window of log lines
- mask some of the window
- predict the masked part in latent space

# data preparation

we need to
1. read log files
2. build a vocabulary (standard for tokenizers)
3. tokenize each log line
4. create a window of log lines
5. create multiple pairs of (masked, unmasked) for each window - call these "samples"

each sample contains:
1. the tokens themselves (log line ran through tokenizer)
2. the token mask (boolean array of padding vs. real tokens)
3. the context mask (what the context encoder sees - this is a masking of the full log line)
4. the target mask (what the predictor should learn to recover, given the context mask)

# model architecture

in JEPA, we have three components:

1. context encoder:
   - the part that sees the visible log context
   - during training, it gets the masked (partial) inpupt, and turns it into a learned representation
   - during inference, this is the part we use to create embeddings for clustering
2. target encoder:
   - the “teacher” branch.
   - looks at the hidden part of the input and produces the target representation that the model should learn to predict
   - not updated directly by gradients; slowly follows the context encoder using EMA
3. predictor:
   - the part that takes the context encoder’s output and tries to guess the target encoder’s hidden representation
   - it uses positional information so it knows which hidden region it is predicting
   - it predicts in latent space
   - the predictor's output is used to compute training loss, which is then used for context encoder's backprop

JEPA trains on multiple masked views of the same log window. The context encoder processes the visible part after masking, the target encoder produces the representation of the hidden part, and the predictor learns to match that hidden representation in latent space.

as such, we need a tokenizer that handles log syntax well, and embeddings that the model learns from our dataset

# what does the model learn

the model learns a reprpesentation of log windows - this representationo is what we cluster on.

the prediction task is just the training signal that makes the encoder actually learn

