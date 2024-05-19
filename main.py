from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import start_http_server
from fastapi import FastAPI, UploadFile, File, Request
from PIL import Image
import io
from tensorflow import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import argparse
import uvicorn
from keras.layers import InputLayer
from scipy import ndimage
import os
import psutil
import time
import request

## Create Registry
custom_registry = CollectorRegistry()

## API 
app = FastAPI()
Instrumentator().instrument(app).expose(app)

## Create metric variables
REQUEST_COUNTER = Counter('api_requests_total', 'Total number of API requests', registry=custom_registry)

RUN_TIME_GAUGE = Gauge('api_run_time_seconds', 'Running time of the API', registry=custom_registry)
TL_TIME_GAUGE = Gauge('api_tl_time_microseconds', 'Effective processing time per character', registry=custom_registry)

MEMORY_USAGE_GAUGE = Gauge('api_memory_usage', 'Memory usage of the API process', registry=custom_registry)
CPU_USAGE_GAUGE = Gauge('api_cpu_usage_percent', 'CPU usage of the API process', registry=custom_registry)

NETWORK_BYTES_SENT_GAUGE = Gauge('api_network_bytes_sent', 'Network bytes sent by the API process', registry=custom_registry)
NETWORK_BYTES_RECV_GAUGE = Gauge('api_network_bytes_received', 'Network bytes received by the API process', registry=custom_registry)


## Parse path to the model from command line
def parse_command():

    parser = argparse.ArgumentParser(description='Load model')
    parser.add_argument('path',type=str, help="Path of the model")
    
    ## parse the arguements
    args = parser.parse_args()

    return args.path

## Load the model
def load(file_path):

    model = load_model(file_path)
    return model

## Format image to the required size
def format_image(image):

    ##convert to grayscale
    img_gray = image.convert('L') 
    ##resize and normalzie
    img_resized = np.array(img_gray.resize((28, 28)))/255.0

    ##center the image
    cy, cx = ndimage.center_of_mass(img_resized)
    rows, cols = img_resized.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    
    ##translate the image
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = ndimage.shift(img_resized, (shifty, shiftx), cval=0)
    
    ##flatten the image
    return img_centered.flatten()

## Get memory allocation
def process_memory():
    return psutil.virtual_memory().used/(1024)

## Predict digit from the input image
def predict_digit(model, img):

    prediction = model.predict(img.reshape(1,-1))

    ##select the max
    return str(max(enumerate(prediction), key=lambda x: x[1])[0])

## MNIST digit prediction
@app.post('/predict')
async def predict(request: Request, file: UploadFile = File(...)):

    ## start time of API call
    start_time = time.time() 
    ## memory usage                  
    memory_usage_start = process_memory()       
    
    client_ip = request.client.host             
    
    # Update network I/O gauges
    network_io_counters = psutil.net_io_counters()

    ## wait till the file is read
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    ## convert to grayscale and normalize
    img_array = format_image(img)

    ## take the path to model through command line and load model
    path = parse_command()
    model = load(path)
    
    ## predict the digit
    pred = predict_digit(model, img_array)

    ## cpu usage
    cpu_percent = psutil.cpu_percent(interval=1) 
    ## memory usage after api call
    memory_usage_end = process_memory()          

    ## set gauages
    CPU_USAGE_GAUGE.set(cpu_percent)                                            
    MEMORY_USAGE_GAUGE.set((np.abs(memory_usage_end-memory_usage_start)))   
    NETWORK_BYTES_SENT_GAUGE.set(network_io_counters.bytes_sent)             
    NETWORK_BYTES_RECV_GAUGE.set(network_io_counters.bytes_recv)            
    
    # Calculate API running time
    end_time = time.time()
    run_time = end_time - start_time         
    REQUEST_COUNTER.labels(client_ip).inc()            
    RUN_TIME_GAUGE.set(run_time)                  
    
    # Calculate T/L time
    input_length = len(contents)
    tl_time = (run_time / input_length) * 1e6  
    TL_TIME_GAUGE.set(tl_time)        
    print(TL_TIME_GAUGE)

    return {"digit": pred}

if __name__ == '__main__':

    ## Start Prometheus metrics server
    start_http_server(8000)
    while True:
    ## Run the FastAPI application
        uvicorn.run(
        "main:app",
        reload=True,
        workers=1,
        host='127.0.0.1',
        port=8080
        )