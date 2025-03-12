from setuptools import setup, find_packages
import os

setup(
    name="eeg_auditory_stimulus",
    version="0.1.0",
    packages=find_packages(include=["eeg_auditory_stimulus", "preprocessing", "eeg_auditory_stimulus.*", "preprocessing.*"]),
    install_requires=[
        "pyyaml", "mne", "pandas", "datetime", "PyQt5", "scikit-learn", "seaborn"
    ],
    author="Nguyen Ha, Khanh Ha, Joobee Jung, Trisha Prasant",
    author_email=["nguyenbh@uw.edu", "bkha@uw.edu", "jbjunguw@uw.edu", "trishp3@uw.edu"],
    description="EEG stimulus-response analysis tool",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    include_package_data=True,
    package_data={
        "eeg_auditory_stimulus": ["configs/*.yml"],  # Include YAML config files
    },
)
