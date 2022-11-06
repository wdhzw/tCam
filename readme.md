## ESP32-based Thermal Imaging Cameras
tCam and tCam-Mini are two cameras I designed around the ESP32 and Lepton 3.5.  They are designed to provide easy access to radiometric data from the Lepton.  Radiometric data is useful because it contains temperature information for each pixel in the camera's image allowing for all kinds of data analysis even if the image is stored to a file.

This repository was created on Nov 6, 2022 from the ESP32 section of my original [Lepton](https://github.com/danjulio/lepton) repository to reduce the size of the portion of that directory that most people are interested in.

### tCam
tCam is a full featured, battery powered camera with a local touchscreen display, local storage and a WiFi interface.  It is comprised of a [gCore]() and tCam-Mini.  gCore can be purchased from Group Gets, along with tCam-Mini.

![tCam](tCam/pictures/tcam_iron.png)

### tCam-Mini
tCam-Mini is a smaller network-only camera designed for streaming and remote access.  It can be built using development boards or a tested unit can be purchased from Group Gets [with built in antenna](https://store.groupgets.com/products/tcam-mini-rev4-wireless-streaming-thermal-camera-board) or [with an external antenna](https://store.groupgets.com/products/tcam-mini-rev4-external-antenna-wireless-streaming-thermal-camera-board).

![tCam-Mini](pictures/tcam_mini.png)
(Photo Credit: Matthew Navarro)

### Desktop Application
Both cameras share a common command set for remote access.  A companion Desktop computer application may be used to configure and access the cameras as well as perform some analysis.

![Desktop Application](pictures/app_on_desktop.png)

### IOS Application
Connect, view and save images on an iPhone or iPad using the [IOS App](https://apps.apple.com/us/app/tcamview/id1610451813).  Documentation may be found on my [website](https://danjuliodesigns.com/products/tcam_app.html).

![IOS App](pictures/ios_app.png)

### Web Server
A simple stand-alone server that can connect to multiple cameras, record images that meet certain criteria and display images from the cameras in a web browser.

![Web Application](pictures/tcam_web.png)

### Python Driver
A simple driver to allow access from python programs.

### R Driver
Bob Rudus wrote a R driver for tCam-Mini that can be found in his [github repository](https://github.com/hrbrmstr/tcam).