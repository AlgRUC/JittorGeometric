#!/usr/bin/env python3
"""
JittorGeometric Web Frontend - Clean Version
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import time
import os
import json

app = Flask(__name__, template_folder='templates')

STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_status.json')

MODEL_DATASETS = {
    'GCN': ['Cora', 'Citeseer', 'Pubmed'],
    'GAT': ['Cora', 'Citeseer', 'Pubmed'],
    'GraphSAGE': ['Cora', 'Citeseer', 'Pubmed'],
    'ChebNet2': ['Cora', 'Citeseer', 'Pubmed'],
    'SGC': ['Cora', 'Citeseer', 'Pubmed'],
    'APPNP': ['Cora', 'Citeseer', 'Pubmed'],
    'GPRGNN': ['Cora', 'Citeseer', 'Pubmed'],
    'EvenNet': ['Cora', 'Citeseer', 'Pubmed'],
    'BernNet': ['Cora', 'Citeseer', 'Pubmed'],
}

def init_status_file():
    if not os.path.exists(STATUS_FILE):
        status = {
            'running': False,
            'epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'acc': 0.0,
            'error': None,
            'history': {'loss': [], 'acc': []},
            'finished': False
        }
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f)
        print(f"[DEBUG] Status file created at: {os.path.abspath(STATUS_FILE)}")
    else:
        print(f"[DEBUG] Status file already exists, keeping it: {os.path.abspath(STATUS_FILE)}")

init_status_file()

def read_training_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
            default_status = {
                'running': False,
                'epoch': 0,
                'total_epochs': 0,
                'loss': 0.0,
                'acc': 0.0,
                'error': None,
                'history': {'loss': [], 'acc': []},
                'finished': False
            }
            for key, value in default_status.items():
                if key not in status:
                    status[key] = value
            if 'history' not in status:
                status['history'] = {'loss': [], 'acc': []}
            if 'loss' not in status['history']:
                status['history']['loss'] = []
            if 'acc' not in status['history']:
                status['history']['acc'] = []
                
            print(f"[DEBUG] Read status from {STATUS_FILE}: epoch={status.get('epoch')}, loss={status.get('loss')}, acc={status.get('acc')}, finished={status.get('finished')}, history_loss_len={len(status.get('history', {}).get('loss', []))}")
            return status
    except Exception as e:
        print(f"[DEBUG] Error reading status: {e}")
        return {
            'running': False,
            'epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'acc': 0.0,
            'error': None,
            'history': {'loss': [], 'acc': []},
            'finished': False
        }

def write_training_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    if not os.path.exists(assets_path):
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    return send_from_directory(assets_path, filename)

@app.route('/api/status')
def status():
    return jsonify(read_training_status())

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json()
    model_name = data.get('model_name', 'GCN')
    dataset_name = data.get('dataset_name', 'Cora')
    epochs = data.get('epochs', 200)
    
    print(f"[DEBUG] Starting training: model={model_name}, dataset={dataset_name}, epochs={epochs}")
    
    training_status = {
        'running': True,
        'epoch': 0,
        'total_epochs': epochs,
        'loss': 0.0,
        'acc': 0.0,
        'error': None,
        'history': {'loss': [], 'acc': []},
        'finished': False
    }
    write_training_status(training_status)
    print(f"[DEBUG] Status file reset for new training")
    
    def start_training():
        import subprocess
        import sys
        
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_worker.py')
        cmd = [
            sys.executable,
            worker_path,
            '--model', model_name,
            '--dataset', dataset_name,
            '--epochs', str(epochs),
            '--hidden_dim', str(data.get('hidden_dim', 16)),
            '--num_layers', str(data.get('num_layers', 2)),
            '--heads', str(data.get('heads', 8)),
            '--dropout', str(data.get('dropout', 0.5))
        ]
        
        print(f"[DEBUG] Starting training worker: {' '.join(cmd)}")
        
        def output_reader(pipe):
            while True:
                line = pipe.readline()
                if not line:
                    break
                print(f"[TRAIN] {line.rstrip()}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        import threading
        reader_thread = threading.Thread(target=output_reader, args=(process.stdout,))
        reader_thread.start()
        
        process.wait()
        reader_thread.join()
        print(f"[DEBUG] Training worker finished with exit code {process.returncode}")
        if process.returncode != 0:
            final_status = read_training_status()
            final_status['error'] = f"Training failed with exit code {process.returncode}"
            write_training_status(final_status)
    
    thread = threading.Thread(target=start_training)
    thread.start()
    
    return jsonify({'status': 'Training started'})

@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    status = read_training_status()
    status['error'] = 'Training stopped by user'
    status['running'] = False
    write_training_status(status)
    return jsonify({'success': True, 'message': 'Training stopped'})

if __name__ == '__main__':
    print(f"[DEBUG] Starting web server on port 5000")
    print(f"[DEBUG] Status file will be at: {os.path.abspath(STATUS_FILE)}")
    app.run(host='0.0.0.0', port=5000, debug=False)
