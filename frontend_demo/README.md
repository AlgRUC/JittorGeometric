# JittorGeometric Web Frontend Demo

A web-based frontend for visualizing training flows in JittorGeometric.

## Features

### Models Supported
- **Graph Neural Networks**: GCN, GAT, GraphSAGE, ChebNet2, SGC, APPNP, GPRGNN, EvenNet, BernNet

### Datasets Supported
- Cora, Citeseer, Pubmed (Planetoid datasets)

### Training Visualization
- Real-time progress bar
- Live loss and accuracy updates
- Interactive charts for training history
- Support for custom training parameters
- Training history persistence (keeps data after training completes)

## Installation

```bash
cd frontend_demo
pip install -r frontend_requirements.txt
```

## Usage

### Start the Web Server

```bash
./start_web_frontend.sh
```

### Stop the Web Server

```bash
./stop_web_frontend.sh
```

### Access the Interface

Open your browser and navigate to the URL provided in the terminal output (typically `http://127.0.0.1:5000`).

## Configuration

The frontend allows you to configure:
- **Model Selection**: Choose from multiple GNN models
- **Dataset Selection**: Select from relevant benchmark datasets (updates based on model)
- **Hidden Dimension**: Adjust model capacity
- **Number of Layers**: Control model depth
- **Heads**: For attention-based models (only visible for GAT)
- **Dropout**: Regularization
- **Epochs**: Training duration

## Files Structure

```
frontend_demo/
├── web_server.py               # Main Flask application
├── train_worker.py             # Training worker with real Jittor training
├── templates/index.html        # Frontend interface
├── frontend_requirements.txt   # Dependencies
├── start_web_frontend.sh       # Startup script
├── stop_web_frontend.sh        # Shutdown script
└── README.md                   # This file
```

## Technical Details

- **Framework**: Flask for web server
- **Visualization**: Chart.js for interactive charts
- **Backend**: JittorGeometric for real model training
- **Architecture**: Client-server with REST API and separate worker process
- **Status**: Training status synchronized via JSON file between processes

## Notes

- The frontend runs on CPU by default to avoid GPU memory conflicts
- All text is in English as required for production
- No operator fusion visualization is included (training flow only)
- Model-dataset compatibility: Dataset options update based on selected model

## License

MIT License

## Contact

For issues or questions, please open a GitHub issue in the JittorGeometric repository.
