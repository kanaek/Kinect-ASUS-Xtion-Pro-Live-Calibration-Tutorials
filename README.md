<!--more-->
# Kinect-ASUS-Xtion-Pro-Live-Calibration-Tutorials
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

### Preparation
#### Installation
* Enter the following codes in your terminal

```bash
sudo apt-get install ros-indigo-openni-camera
sudo apt-get install ros-indigo-openni-launch
```

#### If you are using Asus Xtion Pro Live:
Modify `GlobalDefaults.ini` 
```bash
sudo gedit /etc/openni/GlobalDefaults.ini
```
then uncomment the line: `;UsbInterface=2` (just delete the `;` symbol)

#### If you are using kinect v1.0:
```bash
mkdir ~/kinectdriver 
cd ~/kinectdriver 
git clone https://github.com/avin2/SensorKinect 
cd SensorKinect/Bin/
tar xvjf SensorKinect093-Bin-Linux-x64-v5.1.2.1.tar.bz2
cd Sensor-Bin-Linux-x64-v5.1.2.1/
sudo ./install.sh
```

#### Test your Kinect(Asus Xtion Pro Live) with ROS

* To view the rgb image:
```bash
roslaunch openni_launch openni.launch
rosrun image_view image_view image:=/camera/rgb/image_raw
```

* To visualize the depth_registered point clouds:

```bash
roslaunch openni_launch openni.launch depth_registration:=true
rosrun rviz rviz
```

See [kienct calibration.pdf](https://github.com/CTTC/Kinect-ASUS-Xtion-Pro-Live-Calibration-Tutorials/blob/master/kienct%20calibration.pdf) to view the tutorial of how to calibrate kinect. And note that these codes have been tested on OpenCV 2.4. 
