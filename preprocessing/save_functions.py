import os
import datetime
import matplotlib.pyplot as plt

def save_log(patient_id, base_dir, log_data):
    """Saves log data for a patient with timestamp."""
    filename = "log.txt"
    # patient_folder = os.path.join(base_dir, patient_id)
    
    # os.makedirs(patient_folder, exist_ok=True)
    
    # log_path = os.path.join(patient_folder, filename)
    with open(base_dir, "a") as log_file:  # Open file in append mode
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {log_data}\n")  # Append timestamped log entry
    
def save_plot(patient_id, base_dir, fig, filename):
    """Saves a plot as a PNG file with timestamp."""
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{filename}.png"
    # patient_folder = os.path.join(base_dir, patient_id)
    
    # os.makedirs(patient_folder, exist_ok=True)
    
    plot_path = os.path.join(base_dir, filename)
    fig.savefig(plot_path)
