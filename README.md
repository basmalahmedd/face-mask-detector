# 😷 face-mask-detector

Real-time face mask detection using OpenCV and TensorFlow – detects faces using Haar Cascades and classifies them as "Mask" or "No Mask" using a trained Keras model in real time via webcam.


## 🧠 How It Works

- 📷 Captures real-time webcam video using OpenCV.
- 🧍 Detects faces using Haar Cascade (`haarcascade_frontalface_default.xml`).
- 🧠 The entire frame is passed to a trained CNN model (`mask_detector.h5`) for prediction.
- ✅ Displays a green box for **Mask**, red box for **No Mask**.

## 🧾 Requirements

Install the required packages using pip:

```bash
pip install opencv-python tensorflow keras numpy
```

## 🚀 Run the Project

1. Clone this repository:

```bash
git clone https://github.com/basmalahmedd/face-mask-detector.git
cd face-mask-detector
```

2. Activate your Python environment:

```bash
conda activate mask-detector
```

3. Run the script:

```bash
python mask_detector.py
```

4. Press `ESC` to exit the webcam window.



## 🧪 Model Info

- Input shape: (224, 224, 3)
- Output: Binary classification (`[1] = Mask`, `[0] = No Mask`)
- Training was done using a dataset of face images with and without masks.
The model (mask_detector.h5) was taken from [https://github.com/balajisrinivas](https://github.com/balajisrinivas/Face-Mask-Detection)

## 🎥 Task Submission (Uneeq Interns)

- ✅ Code hosted on GitHub.

## ✨ Credits

- OpenCV: https://opencv.org/
- Keras: https://keras.io/
- TensorFlow: https://www.tensorflow.org/

## 📜 License

MIT License – Feel free to use, modify, and share!
