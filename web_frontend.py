#!/usr/bin/env python3
# Force Jittor to use CPU before importing
import os
os.environ['JITTOR_CPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

"""
Web frontend for JittorGeometric using Flask
"""

from flask import Flask, render_template, request, jsonify
from model_training_demo import ModelTrainer
import threading
import time

app = Flask(__name__)

# Global variables for training status
training_status = {
    'running': False,
    'epoch': 0,
    'loss': 0.0,
    'acc': 0.0,
    'total_epochs': 0,
    'start_time': 0.0
}

trainer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    models = ['GCN', 'GAT', 'GraphSAGE']
    return jsonify(models)

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    return jsonify(datasets)

@app.route('/api/train', methods=['POST'])
def train():
    global training_status, trainer
    
    if training_status['running']:
        return jsonify({'error': 'Training already running'}), 400
    
    data = request.get_json()
    model_name = data.get('model_name', 'GCN')
    dataset_name = data.get('dataset_name', 'Cora')
    hidden_dim = data.get('hidden_dim', 16)
    num_layers = data.get('num_layers', 2)
    heads = data.get('heads', 8)
    dropout = data.get('dropout', 0.5)
    epochs = data.get('epochs', 200)
    
    # Initialize trainer
    trainer = ModelTrainer(model_name=model_name, dataset_name=dataset_name)
    trainer.load_data()
    trainer.create_model(hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout)
    
    # Start training in a separate thread
    training_status = {
        'running': True,
        'epoch': 0,
        'loss': 0.0,
        'acc': 0.0,
        'total_epochs': epochs,
        'start_time': time.time()
    }
    
    def train_thread():
        global training_status
        try:
            trainer.train(epochs=epochs, status_callback=lambda epoch, loss, acc: update_training_status(epoch, loss, acc))
            acc = trainer.test()
            training_status['acc'] = acc
            training_status['running'] = False
        except Exception as e:
            training_status['error'] = str(e)
            training_status['running'] = False
    
    threading.Thread(target=train_thread).start()
    
    return jsonify({'message': 'Training started'})

def update_training_status(epoch, loss, acc):
    global training_status
    training_status['epoch'] = epoch
    training_status['loss'] = loss
    training_status['acc'] = acc

@app.route('/api/status', methods=['GET'])
def get_status():
    global training_status
    return jsonify(training_status)

@app.route('/api/predict', methods=['POST'])
def predict():
    global trainer
    
    if not trainer:
        return jsonify({'error': 'No model trained yet'}), 400
    
    data = request.get_json()
    x = data.get('x')
    edge_index = data.get('edge_index')
    
    if not x or not edge_index:
        return jsonify({'error': 'Missing x or edge_index'}), 400
    
    # Convert to jittor tensors
    import jittor as jt
    x = jt.array(x)
    edge_index = jt.array(edge_index)
    
    # Predict
    pred = trainer.predict(x, edge_index)
    pred = pred.argmax(dim=1).numpy().tolist()
    
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    # Force Jittor to use CPU
    import os
    os.environ['JITTOR_CPU'] = '1'
    
    # Create templates directory if not exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JittorGeometric Web Frontend</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, input[type="number"], button {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
        }
        .status.running {
            background-color: #e3f2fd;
            color: #1565c0;
        }
        .status.completed {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .status.error {
            background-color: #ffebee;
            color: #c62828;
        }
        .progress {
            width: 100%;
            height: 20px;
            background-color: #ddd;
            border-radius: 10px;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>JittorGeometric Web Frontend</h1>
        
        <div class="form-group">
            <label for="model_name">Model:</label>
            <select id="model_name">
                <option value="GCN">GCN</option>
                <option value="GAT">GAT</option>
                <option value="GraphSAGE">GraphSAGE</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="dataset_name">Dataset:</label>
            <select id="dataset_name">
                <option value="Cora">Cora</option>
                <option value="Citeseer">Citeseer</option>
                <option value="Pubmed">Pubmed</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="hidden_dim">Hidden Dimension:</label>
            <input type="number" id="hidden_dim" value="16" min="1" max="256">
        </div>
        
        <div class="form-group">
            <label for="num_layers">Number of Layers:</label>
            <input type="number" id="num_layers" value="2" min="1" max="10">
        </div>
        
        <div class="form-group">
            <label for="heads">Heads (for GAT only):</label>
            <input type="number" id="heads" value="8" min="1" max="16">
        </div>
        
        <div class="form-group">
            <label for="dropout">Dropout:</label>
            <input type="number" id="dropout" value="0.5" min="0.0" max="1.0" step="0.1">
        </div>
        
        <div class="form-group">
            <label for="epochs">Epochs:</label>
            <input type="number" id="epochs" value="200" min="1" max="1000">
        </div>
        
        <button id="train_btn">Start Training</button>
        
        <div id="status" class="status"></div>
        
        <div id="progress" class="progress">
            <div id="progress_bar" class="progress-bar" style="width: 0%"></div>
        </div>
    </div>
    
    <script>
        const trainBtn = document.getElementById('train_btn');
        const statusDiv = document.getElementById('status');
        const progressBar = document.getElementById('progress_bar');
        
        trainBtn.addEventListener('click', async () => {
            trainBtn.disabled = true;
            statusDiv.innerHTML = 'Starting training...';
            statusDiv.className = 'status running';
            
            const data = {
                model_name: document.getElementById('model_name').value,
                dataset_name: document.getElementById('dataset_name').value,
                hidden_dim: parseInt(document.getElementById('hidden_dim').value),
                num_layers: parseInt(document.getElementById('num_layers').value),
                heads: parseInt(document.getElementById('heads').value),
                dropout: parseFloat(document.getElementById('dropout').value),
                epochs: parseInt(document.getElementById('epochs').value)
            };
            
            try {
                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error('Training failed to start');
                }
                
                // Poll status
                const interval = setInterval(async () => {
                    const statusResponse = await fetch('/api/status');
                    const status = await statusResponse.json();
                    
                    if (status.running) {
                        statusDiv.innerHTML = `Training running... Epoch ${status.epoch}/${status.total_epochs}, Loss: ${status.loss.toFixed(4)}, Acc: ${status.acc.toFixed(4)}`;
                        const progress = (status.epoch / status.total_epochs) * 100;
                        progressBar.style.width = `${progress}%`;
                    } else {
                        clearInterval(interval);
                        trainBtn.disabled = false;
                        
                        if (status.error) {
                            statusDiv.innerHTML = `Error: ${status.error}`;
                            statusDiv.className = 'status error';
                        } else {
                            statusDiv.innerHTML = `Training completed. Test accuracy: ${status.acc.toFixed(4)}`;
                            statusDiv.className = 'status completed';
                            progressBar.style.width = '100%';
                        }
                    }
                }, 1000);
                
            } catch (error) {
                trainBtn.disabled = false;
                statusDiv.innerHTML = `Error: ${error.message}`;
                statusDiv.className = 'status error';
            }
        });
    </script>
</body>
</html>
''')
    
    # Find an available port
    import socket
    def find_available_port(start_port=5000, end_port=6000):
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return None
    
    port = find_available_port()
    if port is None:
        print('No available ports found')
    else:
        print(f'Found available port: {port}')
        print(f'Local access: http://127.0.0.1:{port}')
        print(f'Remote access: http://219.216.65.209:{port}')
        app.run(host='0.0.0.0', port=port, debug=False)