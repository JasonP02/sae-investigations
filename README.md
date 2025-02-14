# SAE Investigations

Research project for investigating Sparse Autoencoder (SAE) behavior in language models.

## Project Structure

```
sae-investigations/
├── data/               # Dataset storage
├── notebooks/         # Jupyter notebooks for analysis
├── outputs/           # Generated outputs and results
├── src/              # Source code
│   ├── core/         # Core functionality
│   │   ├── config.py           # Configuration classes
│   │   ├── models.py           # Data models and containers
│   │   ├── text_generation.py  # Text generation logic
│   │   ├── experiment_runner.py # Experiment execution
│   │   └── model_setup.py      # Model initialization
│   ├── evaluation/   # Evaluation scripts
│   │   └── interactive_testing.py # Interactive testing
│   ├── utils/       # Utility functions
│   └── visualization/ # Visualization tools
│       └── visualizers.py      # Visualization functions
├── experiments/      # Experiment results
└── tests/           # Test cases
```

## Core Components

- **Text Generation**: Handles token generation with SAE analysis capabilities
- **Experiment Runner**: Manages experiment execution and data collection
- **Visualization**: Tools for analyzing and visualizing results
- **Evaluation**: Scripts for model evaluation and testing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run interactive testing:
```bash
python src/evaluation/interactive_testing.py
```

## Configuration

The project uses a centralized configuration system (`src/core/config.py`) for managing:
- Generation parameters
- Model settings
- Experiment configurations
- Visualization options

See `config.py` for detailed parameter descriptions. 