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

* Remember to modify `GlobalDefaults.ini` if you are using Asus Xtion Pro Live
```bash
sudo gedit /etc/openni/GlobalDefaults.ini
```
then uncomment the line: `;UsbInterface=2` (just delete the `;` symbol)

#### Test your Asus Xtion Pro Live with ROS

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

See 'kienct calibration.pdf' to view the tutorial of how to calibrate kinect.
