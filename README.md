# Eagle-Vision CCTV

Eagle-Vision CCTV is a surveillance system engineered to make any CCTV camera into an advanced AI detection technology. It utilizes YOLO (You Only Look Once) for real-time object detection, offering a robust solution for home security, business surveillance, and public safety efforts.

## Features

- **Real-time Video Streaming:** Monitor your surroundings in real-time with live feeds.
- **AI-Powered Object Detection:** Leverage YOLO for precise detection of people and objects with customizable confidence thresholds.
- **Instant Notifications:** Receive immediate Telegram alerts when a person is detected, enabling swift action.
- **Automated Event Recording:** Events are automatically saved as video clips and thumbnails in a gallery for easy access and evidence collection.
- **Customizable Settings:** Adjust camera inputs, detection sensitivity, and notification preferences via an intuitive settings page.
- **Cross-Platform Support:** Built with Python for wide-ranging compatibility across different operating systems.

## Getting Started

### Prerequisites

Before beginning the installation of Eagle-Vision CCTV, you'll need to have Conda installed on your system. Conda is an open-source package management system and environment management system that can run on Windows, macOS, and Linux. It simplifies package management and deployment for Python applications.

**If you do not have Conda installed:**

1. Visit the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) website. Miniconda is a minimal installer for Conda, while Anaconda is a distribution that includes Conda and a large number of Python packages, including scientific computing, data science, and machine learning libraries.

### Installation

1. **Create and Activate a Conda Environment:**
    First, you need to create a dedicated Conda environment for Eagle-Vision CCTV. Open a terminal and run the following commands:
    ```bash
    conda create --name eagle-vision python=3.10
    conda activate eagle-vision
    ```

2. **Clone the Repository:**
    With your Conda environment activated, clone the Eagle-Vision CCTV repository to your local machine:
    ```bash
    git clone https://github.com/jjiteshh/eagle-vision-cctv.git
    cd eagle-vision-cctv
    ```

3. **Install Required Dependencies:**
    Install all the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. **Start the Application:**
    Finally, to start the Eagle-Vision CCTV application, run:
    ```bash
    python main.py
    ```

After completing these steps, Eagle-Vision CCTV should be up and running, monitoring your designated areas with real-time AI-powered object detection.
