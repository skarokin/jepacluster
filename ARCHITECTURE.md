# jepacluster Architecture

This is a focused guide to the current codebase.
It explains:

1. what each file does
2. what each class does
3. what each function does
4. how the training loop works
5. how this differs from the official `facebookresearch/jepa` repo

I’m assuming you know basic ML ideas, but not JEPA specifically.

---

## 1. What this project is trying to do

`jepacluster` is a log clustering system inspired by JEPA.

The idea is:

- read a window of log lines
- hide some of the window
- predict the hidden part in latent space
- use the learned representation to cluster logs later

So the model is not trying to memorize exact text.
It is trying to learn the structure of log behavior.

---

## 2. Files and what they do

### `src/jepacluster/main.py`

This is the entrypoint.
It parses CLI flags, loads the config, and starts either training or inference.

### `src/jepacluster/utils/config.py`

This defines the configuration schema.
It makes sure `config.yaml` has the right shape and data types.

### `src/jepacluster/core/dataset.py`

This loads log text, tokenizes it, builds a vocabulary, creates windows, and creates masks.
It turns raw logs into model-ready tensors.

### `src/jepacluster/core/architecture.py`

This defines the model:

- token embedding
- log sequence encoder
- predictor
- target encoder copy

### `src/jepacluster/core/trainer.py`

This runs training.
It does the forward pass, computes the JEPA loss, applies gradient updates, updates the target encoder with EMA, logs collapse statistics, and saves checkpoints.

### `src/jepacluster/engine/embedder.py`

This is meant to load the trained model and create embeddings for inference.

### `src/jepacluster/engine/clusterer.py`

This is meant to cluster embeddings, usually with UMAP + HDBSCAN.

### `src/jepacluster/engine/analyzer.py`

This is meant to interpret the clusters after clustering.

---

## 3. Classes and what they do

## 3.1 `utils/config.py`

### `UMAPConfig`
Stores UMAP parameters:

- `n_components`
- `n_neighbors`
- `min_dist`

### `HDBSCANConfig`
Stores HDBSCAN parameters:

- `min_cluster_size`
- `min_samples`
- `prediction_threshold`

### `ClusteringConfig`
Groups UMAP and HDBSCAN configs together.

### `TrainingConfig`
Stores training parameters:

- batch size
- learning rate
- epochs
- EMA decay
- loss type
- vocabulary cutoff
- weight decay
- warmup
- gradient clipping
- AMP usage
- resume checkpoint path

### `EncoderConfig`
Stores encoder settings:

- number of layers
- number of heads
- dropout

### `PredictorConfig`
Stores predictor settings:

- hidden dimension
- predictor layers
- predictor heads

### `ModelConfig`
Stores model settings:

- name
- window size
- latent dimension
- encoder config
- predictor config
- vocab size
- tokenizer path

### `Config`
Top-level config wrapper.

Its job is to hold:

- model config
- training config
- clustering config

---

## 3.2 `dataset.py`

### `LogMaskDataset`
This is the dataset class.

It:

- reads log files
- builds a vocabulary
- tokenizes log lines
- creates windows of log lines
- creates multiple mask pairs
- returns sample dictionaries

Each sample contains:

- `tokens`
- `token_mask`
- `context_mask`
- `target_mask`

### `MaskSampler`
This creates several different masked views of the same window.

Why?

Because JEPA learns better when it sees multiple hidden spans, not just one fixed one.

### `collate_jepa_batches`
This stacks dataset samples into a single batch for the model to process together.

This is necessary since the dataset gives one sample at a time (one log window, one token mask, one context mask, one target mask), but the trainer should feed multiple of these to the GPU

---

## 3.3 `architecture.py`

### `TokenEmbedder`
Turns token ids into learned vectors.

### `FeedForwardBlock`
A small neural block used after the transformer layers.

### `LogSequenceEncoder`
The main encoder for logs.

It:

- embeds token ids
- adds positional information
- collapses token-level information into line-level information
- uses transformer layers
- applies a feed-forward block
- normalizes the output

### `Predictor`
Takes context embeddings and target positions and predicts hidden latent vectors.

### `JEPAArchitecture`
Wraps the whole JEPA setup:

- context encoder
- target encoder
- predictor

It also:

- copies context weights into target weights at initialization
- freezes target gradients
- updates the target encoder with EMA

---

## 3.4 `trainer.py`

### `Trainer`
This class contains the training loop.

It:

- loads the dataset
- creates the model
- creates the optimizer
- creates the iteration-based schedule
- samples masks
- computes loss
- updates weights
- updates the target encoder
- saves checkpoints
- resumes from checkpoints

---

## 4. Functions and what they do

## 4.1 `main.py`

### `parse_args()`
Defines the CLI arguments.

### `main()`
Loads config, decides train vs infer, and calls the right classes.

---

## 4.2 `config.py`

### `validate_config(config_file)`
Reads YAML from disk and validates it using Pydantic.

---

## 4.3 `dataset.py`

### `LogMaskDataset.__init__`
Sets up paths, vocab settings, tokenizer settings, and builds samples.

### `_build_vocab()`
Scans training logs and builds the token vocabulary.

### `_load_sentencepiece()`
Loads a SentencePiece tokenizer if one is available.

### `_tokenize(text)`
Splits text into words, numbers, and punctuation.

### `load_samples()`
Creates training samples from sliding log windows.

### `_build_masks()`
Builds several context/target span pairs.

### `_encode_window(window)`
Tokenizes each log line and stacks them into a tensor.

### `_encode_line(line)`
Turns a single log line into token ids.

### `__len__()`
Returns the number of training samples.

### `__getitem__(idx)`
Returns one sample.

### `collate_jepa_batches(batch)`
Stacks a batch of samples into batch tensors.

### `MaskSampler.sample()`
Generates several mask pairs for one log window.

---

## 4.4 `architecture.py`

### `TokenEmbedder.__init__`
Creates the embedding table.

### `TokenEmbedder.forward(x)`
Looks up embeddings for token ids.

### `FeedForwardBlock.__init__`
Builds the feed-forward subnetwork.

### `FeedForwardBlock.forward(x)`
Runs the feed-forward network.

### `LogSequenceEncoder.__init__`
Builds the log encoder.

### `LogSequenceEncoder.forward(x, mask)`
Embeds token ids, adds positional information, applies the mask, then runs the transformer stack.

### `Predictor.__init__`
Builds the predictor network.

### `Predictor.forward(context_tokens, target_positions)`
Predicts hidden latent vectors for masked positions.

### `JEPAArchitecture.__init__`
Creates the context encoder, target encoder, and predictor.

### `JEPAArchitecture.encode_context(x, mask)`
Runs the context encoder.

### `JEPAArchitecture.encode_target(x, mask)`
Runs the target encoder.

### `JEPAArchitecture.predict(context_latent, target_positions)`
Runs the predictor.

### `JEPAArchitecture.update_target_encoder(ema_decay)`
Updates the target encoder with EMA.

---

## 4.5 `trainer.py`

### `Trainer.__init__`
Sets up:

- device
- dataset
- model
- AMP scaler
- checkpoint state

### `Trainer.train()`
Runs the whole training loop.

Step-by-step:

1. Load a batch.
2. Sample multiple masks.
3. Encode the visible context.
4. Encode the hidden target with the frozen teacher.
5. Predict the hidden target in latent space.
6. Compute smooth L1 loss.
7. Backprop through context encoder and predictor.
8. Update the optimizer.
9. Update the target encoder with EMA.
10. Log latent variance.
11. Save the checkpoint at the end of each epoch.

### `Trainer._build_schedulers(optimizer)`
Sets up iteration-based learning-rate behavior.

### `Trainer._step_schedulers()`
Updates learning rate and weight decay for the current iteration.

### `Trainer._log_collapse_stats(batch, loss)`
Logs latent variance so you can catch collapse early.

### `Trainer.load_data()`
Creates the dataset.

### `Trainer.resume_if_available()`
Loads a checkpoint if the user supplied one.

### `Trainer.save_model(epoch)`
Writes the model checkpoint to disk.

---

## 5. The training loop in plain English

This is the part people usually care about most.

### What happens during one training step

1. The trainer loads a batch of tokenized log windows.
2. It creates several context/target masks.
3. The context encoder sees the visible part.
4. The target encoder sees the hidden part.
5. The predictor tries to guess the hidden latent vectors.
6. The loss compares prediction to teacher output.
7. The context encoder and predictor are updated with backprop.
8. The target encoder is updated with EMA.

### Why this is JEPA-like

JEPA does not try to reconstruct the raw input directly.
It tries to predict hidden representations.

That is why the model learns useful structure instead of just copying text.

---

## 6. Architectural decisions

### Why a context encoder and a target encoder?

The context encoder learns from visible input.
The target encoder provides a stable teacher signal.

This separation helps prevent collapse and keeps training stable.

### Why use EMA for the target encoder?

EMA means “exponential moving average.”

Instead of learning the target encoder directly, we slowly move it toward the context encoder.
This is a standard JEPA/BYOL-style stabilization trick.

### Why use positional embeddings?

Because the predictor needs to know **where** something is missing.
Without position, the model cannot know which region it is predicting.

### Why use masks?

Masks force the model to infer missing information from context.
That is the core self-supervised signal in JEPA-style training.

### Why predict latent vectors instead of text?

Predicting latent vectors is often more stable and more semantically meaningful than predicting raw tokens.

For clustering, we care about the representation space, not exact reconstruction.

---

## 7. Drift from the official `facebookresearch/jepa` repo

### Drift caused by logs vs. images

These differences are expected and appropriate:

- logs are text, not images/videos
- we tokenize text instead of patching pixels
- masks are token spans instead of image blocks
- the downstream task is clustering, not vision probing

### Drift caused by implementation maturity

These are still simpler than the official repo:

- no distributed training stack
- no full mask-collator subsystem
- no full production inference stack
- no full deployment packaging

So this code follows the JEPA idea, but it is adapted to logs and still simpler than Meta’s full research implementation.

---

## 8. Short summary

If you only remember one thing:

`jepacluster` learns to predict hidden log information from visible log context, and then uses the learned representation to cluster logs.

The main moving parts are:

- dataset: turns logs into tokens and masks
- architecture: learns latent representations
- trainer: runs JEPA-style learning
- engine: uses embeddings for clustering

