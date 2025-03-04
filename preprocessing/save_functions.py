import os
import datetime
import matplotlib.pyplot as plt

def save_log(patient_id, base_dir, log_data):
    """Saves log data for a patient with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log_{timestamp}.txt"
    patient_folder = os.path.join(base_dir, patient_id)
    
    os.makedirs(patient_folder, exist_ok=True)
    
    log_path = os.path.join(patient_folder, filename)
    with open(log_path, "w") as log_file:
        log_file.write(log_data)
    
    return log_path

def save_plot(patient_id, base_dir, fig):
    """Saves a plot as a PNG file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_{timestamp}.png"
    patient_folder = os.path.join(base_dir, patient_id)
    
    os.makedirs(patient_folder, exist_ok=True)
    
    plot_path = os.path.join(patient_folder, filename)
    fig.savefig(plot_path)
    
    return plot_path
