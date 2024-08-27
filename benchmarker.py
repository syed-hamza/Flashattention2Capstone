import time
import psutil
import GPUtil
import csv
from threading import Thread
import matplotlib.pyplot as plt
import pandas as pd
import os

class Benchmarker:
    def __init__(self, filename, interval=1):
        self.filename = filename
        self.interval = interval
        self.stop_logging = False
        self.data = []
        self.current_loss = 0

    def log_data(self):
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'CPU Usage (%)', 'RAM Usage (MB)', 'GPU Usage (%)', 'GPU Memory (MB)', 'Loss'])
            
            start_time = time.time()
            while not self.stop_logging:
                timestamp = time.time() - start_time
                cpu_percent = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUsed
                else:
                    gpu_usage = 0
                    gpu_memory = 0
                
                row = [timestamp, cpu_percent, ram_usage, gpu_usage, gpu_memory, self.current_loss]
                writer.writerow(row)
                self.data.append(row)
                
                time.sleep(self.interval)

    def start(self):
        self.thread = Thread(target=self.log_data)
        self.thread.start()

    def stop(self):
        self.stop_logging = True
        self.thread.join()

    def update_loss(self, loss):
        self.current_loss = loss

    def plot_benchmark(self, output_dir):
        df = pd.read_csv(self.filename)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        ax1.plot(df['Timestamp'], df['CPU Usage (%)'], label='CPU')
        ax1.plot(df['Timestamp'], df['GPU Usage (%)'], label='GPU')
        ax1.set_ylabel('Usage (%)')
        ax1.set_title('CPU and GPU Usage Over Time')
        ax1.legend()
        
        ax2.plot(df['Timestamp'], df['RAM Usage (MB)'], label='RAM')
        ax2.plot(df['Timestamp'], df['GPU Memory (MB)'], label='GPU Memory')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Over Time')
        ax2.legend()
        
        ax3.plot(df['Timestamp'], df['Loss'], label='Loss')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss Over Time')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpt_benchmark_plot.png'))
        plt.close()