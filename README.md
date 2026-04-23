# jepacluster

An experimental **Joint-Embedding Predictive Architecture (JEPA)** approach to log clustering. 

Unlike traditional methods that group logs based on syntax, `jepacluster` groups logs based on their **future consequence**. It learns a world model of your system logs, where the similarity between two logs is defined by the similarity of the future system states they will lead to.

`jepacluster` utilizes a sliding window of preceding logs to generate embeddings. This ensures that clusters represent stateful transitions, rather than isolated strings. A log's position in latent space is determined by the trajectory that led to it, and the future it predicts.

So, `jepacluster` takes preceding logs, a target log, and groups the target log into a cluster.

## Example

**Target Log:** `[INFO] Connection closed`
| Preceding Sequence | Target Log | Cluster |
| :--- | :--- | :--- |
| `[INFO] Auth success` -> `[INFO] Data stream end` | `[INFO] Connection closed` | "Graceful Shutdown" |
| `[ERROR] Timeout` -> `[WARN] Retry limit reached` | `[INFO] Connection closed` | "Resource Exhaustion" |

Standard clustering would see the string `Connection closed` and put both into a single "Network Info" bucket.

`jepacluster` would recognize that the first sequence predicts a clean exit state, while the second predicts an unhealthy state. It separates them into different clusters because their **causal trajectories** have nothing in common.

## How it Works

1. **Training (The World Model):** The model watches sequences of logs. It takes a context ($x$) and attempts to predict the latent representation of the future ($y$).
2. **The Encoder:** Through this predictive task, the encoder learns to compress logs into vectors that encode **future system state**.
3. **Inference:** Live logs are passed through the trained encoder. 
4. **Clustering:** We run **HDBSCAN** on the resulting vectors.

## Usage

1. Put training logs in `data/` as `.txt` files. The loader searches recursively, so you can organize logs into subfolders if needed.
2. Adjust `config.yaml` for your dataset. The main knobs are `window_size`, `latent_dim`, the training hyperparameters, and the UMAP/HDBSCAN clustering settings.
3. Train the model from the project root:

```bash
python src/jepacluster/main.py --train --data_dir data/ --model_dir models/ --config_file config.yaml
```

4. Trained artifacts are saved in `models/`.
5. Run inference with the same entrypoint and the saved model directory:

```bash
python src/jepacluster/main.py --infer --infer_dir infer_data/ --model_dir models/ --config_file config.yaml
```
