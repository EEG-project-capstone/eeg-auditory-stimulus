import os
import datetime
import matplotlib.pyplot as plt

def save_log(base_dir, log_data):
    """Saves log data for a patient"""
    filename = "log.txt"

    log_path = os.path.join(base_dir, filename)
    with open(log_path, "a") as log_file:  # Open file in append mode
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{log_data}\n")  # Append timestamped log entry
    
def save_plot(base_dir, fig, filename):
    """Saves a plot as a PNG file"""
    plot_path = os.path.join(base_dir, filename)
    fig.savefig(plot_path)