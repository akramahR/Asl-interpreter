
# ASL Interpreter

This project is an American Sign Language (ASL) interpreter that utilizes video input to recognize and translate ASL gestures in real time. The model is built using a combination of deep learning techniques and video processing.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/asl-interpreter.git
   cd asl-interpreter
   ```

2. **Install the required dependencies:**

   Make sure you have Python 3.6+ and install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Download Video Data:**  
   Run the `youtube_video_downloader.py` script located in the `scripts` directory to download ASL videos from YouTube.

   ```bash
   python scripts/youtube_video_downloader.py
   ```

2. **Download Kinetics-600 Model:**  
   Download the Kinetics-600 UniFormer-B 32x1x4 model from one of the following sources:

   - From the Sense-X GitHub repository: [Kinetics-600 UniFormer-B 32x1x4](https://github.com/Sense-X/UniFormer/tree/main/video_classification)
   - From Google Drive: [Download Here](https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing)

   Place the downloaded model file in the `scripts` directory.

3. **Extract Features:**  
   Run the `ExtractFeatures.py` script to extract features from the downloaded videos.

   ```bash
   python scripts/ExtractFeatures.py
   ```

4. **Train, Test, and Tune the Model:**  
   Run the `trainTestTune.py` script to train, test, and tune the model with the extracted features.

   ```bash
   python scripts/trainTestTune.py
   ```

5. **Real-time Prediction:**  
   For real-time prediction, run the `realtime.py` script in the `scripts` directory.

   ```bash
   python scripts/realtime.py
   ```

## File Structure

```
asl-interpreter/
│
├── scripts/
│   ├── youtube_video_downloader.py
│   ├── ExtractFeatures.py
│   ├── trainTestTune.py
│   └── realtime.py
│
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you would like to add.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
