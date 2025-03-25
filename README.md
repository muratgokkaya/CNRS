# CNRS
Container Number Recognition System

CNRS (Container Number Recognition System). This powerful yet lightweight tool is designed to streamline operations at entry points, loading/unloading zones, and beyond—making logistics faster, smarter, and more efficient.
 
🔍 Key Features & Highlights:

✅ Over 99% Accuracy: High-precision container number detection ensures reliable and error-free operations, even in challenging conditions like rain, cloudy weather, or low light.

✅ No Paid Dependencies: Fully optimized with zero reliance on costly third-party libraries.

✅ Real-Time Processing: Seamlessly supports live RTSP camera feeds without requiring a GPU.

✅ Customizable & Insightful: Includes features like Recognition Region Selection and detailed reporting for deeper insights.

✅ On-Premise & Lightweight: Runs locally, even on Single Board Computers, offering full control, enhanced security, and minimal resource usage.

Install dependencies ;
sudo apt install python3-dev python3.11-dev v4l-utils libcairo2-dev libxt-dev cmake gir1.2-gtk-3.0 libgtk2.0-dev pkg-config gdal-bin libgdal-dev libgirepository1.0-dev -y

Create Python Virtual Env and Run main.py file in that Env.

python3 -m venv ocrenv
source ocrenv/bin/activate

pip install pycoral tflite-runtime
pip install pycoral
pip install opencv-python-headless 
pip install ultralytics 
pip install mjpeg_streamer 
pip install imutils
pip install pytz
pip install flask
pip install PyGObject

python3 main.py




