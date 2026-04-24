# jepacluster/src/jepacluster/utils

Supporting logic for data handling and system configuration.

- **parser.py**: Log pre-processor. Optionally strips high-entropy characters (regex) before tokenization to reduce initial vocabulary size.
- **config.py**: Pydantic-based configuration management. Validates `config.yaml` parameters such as window size, latent dimensions, and clustering thresholds.
- **logger.py**: Shared logger