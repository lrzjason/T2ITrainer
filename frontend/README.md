# LoRA Trainer Node Editor

A visual node-based editor for constructing complex configuration files and commands for LoRA training scripts (specifically for `train_longcat.py` and related scripts).

## Configuration

The application can be configured using environment variables. Copy `.env.example` to `.env` and modify the values as needed:

```bash
cp .env.example .env
```

Environment variables:
- `VITE_API_URL`: The base URL for API requests (default: http://127.0.0.1:8000)
- `VITE_WS_URL`: The WebSocket URL for real-time training output (default: ws://127.0.0.1:8000/ws/training)
- `VITE_GEMINI_API_KEY`: Optional API key for AI features (if applicable)

## Node Types

### Core Nodes
These nodes define the global environment and hyperparameters for the training session.

- **Script Loader**: The main output node.
    - **Inputs**: Directory, LoRA, Misc settings, and the Dataset List.
    - **Function**: Generates the final CLI command (`python train_....py ...`) and the JSON configuration content.
- **Directory Setup**: Defines paths for the pretrained model, output directory, logging, and VAE.
- **LoRA Config**: Network specific settings including Rank, Alpha, LoKR, and RoPE scaling.
- **Misc Settings**: Training hyperparameters like Batch Size, Epochs, Learning Rate, Optimizer, and advanced settings (Min-SNR, NLN).

### Structure Nodes
These nodes organize how data is ingested.

- **Dataset List**: An aggregator that collects multiple `Dataset Config` nodes. Connects to the **Script Loader**.
- **Dataset Config**: Represents a single training dataset block.
    - **Inputs**: Accepts lists for Images, Targets, References, Captions, and Batches.
    - **Settings**: Defines the image folder (`train_data_dir`), resolution, and repeats.

### Component Nodes
These nodes define the granular rules for how data is processed within a dataset.

- **Lists (Image, Target, Caption, Batch, Reference)**: Middleware nodes that collect multiple items of the same type and pass them to the `Dataset Config`.
- **Image Item**: Defines image processing rules (e.g., `suffix` for image files).
- **Target Item**: Defines target image pairs for training (e.g., mapping a `train` image to a specific target).
- **Caption Item**: Defines caption file rules (extension, dropout) and optional Reference Lists for captions.
- **Batch Item**: Links Target, Caption, and Reference configs together for specific batching strategies.

### Reference Nodes (Special)
The Reference configuration is more complex and splits into two parts:

- **Ref Group**: Represents a specific key (e.g., `train_ref`) and acts as a list for multiple sources.
- **Ref Source**: Represents a single source of reference images (e.g., from a subdirectory or same name). Multiple sources connect to one **Ref Group**.

## Workflow

1. **Global Setup**: Connect `Directory`, `LoRA`, and `Misc` nodes to the `Script Loader`.
2. **Data Structure**: Create a `Dataset List` and connect it to `Script Loader`.
3. **Dataset Definition**: Create a `Dataset Config` and connect it to the `Dataset List`.
4. **Component Definition**:
    - Create specific lists (e.g., `Image List`).
    - Create items (e.g., `Image Item`) and connect them to their respective lists.
    - Connect the lists to the `Dataset Config`.
5. **Export**: Use the **Output Preview** panel to copy the JSON config and the Python command.
