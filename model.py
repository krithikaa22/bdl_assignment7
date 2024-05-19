import mlflow
from mlflow.tracking import MlflowClient
from keras.models import load_model, save_model
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

client = MlflowClient(tracking_uri='http://127.0.0.1:5000/') 
experiments = client.search_experiments()


experiment_id="468194401570234112"
runs = mlflow.search_runs(experiment_id)
print(runs)
loss = float('inf')
run_id = None

for run in runs.iterrows():
    l = run[1]['metrics.val_loss']

    if l < loss:
        loss = l
        run_id = run[1]['run_id']
print(run_id)
best_model = mlflow.keras.load_model("runs:/" + run_id + "/model")

os.makedirs('model',exist_ok=True)
save_path = os.path.join('model','MNIST_model')

save_model(best_model,save_path)