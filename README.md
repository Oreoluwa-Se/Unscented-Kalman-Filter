# SFND_Unscented_Kalman_Filter
Sensor Fusion UKF Highway Project

<img src="media/ukf_highway_tracked.gif" width="700" height="400" />

In this project you will implement an Unscented Kalman Filter to estimate the state of multiple cars on a highway using noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric. 

The main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ukf_highway

Note that the programs that need to be written to accomplish the project are src/ukf.cpp, and src/ukf.h

The program main.cpp has already been filled out, but feel free to modify it.

<img src="media/ukf_highway.png" width="700" height="400" />

`main.cpp` is using `highway.h` to create a straight 3 lane highway environment with 3 traffic cars and the main ego car at the center. 
The viewer scene is centered around the ego car and the coordinate system is relative to the ego car as well. The ego car is green while the 
other traffic cars are blue. The traffic cars will be accelerating and altering their steering to change lanes. Each of the traffic car's has
it's own UKF object generated for it, and will update each indidual one during every time step. 

The red spheres above cars represent the (x,y) lidar detection and the purple lines show the radar measurements with the velocity magnitude along the detected angle. The Z axis is not taken into account for tracking, so you are only tracking along the X/Y axis.

---

## Other Important Dependencies
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
 * PCL 1.2

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./ukf_highway`

## Project Directory:
The project directory should be similar to:
```
├── build                           # build files for program
│   ├── ...
│── media                            # stores video output from project                   
├── src                     
│   ├── Eigen                      # helper functions for program
│   ├── render                      # simulates the PI controller
│   │   ├── box.h                   # stores structs for box definition
│   │   ├── render.cpp              # function used for image display
│   │   ├── render.h                # header file for rendering images   
│   ├── sensors 					# stores lidar and radar data
│   ├── highway.cpp 					# logic for handling and creating highway traffic
│   ├── measurement_pacakge.cpp 					# contains measurment types
│   ├── tools.cpp  
│   ├── tools.h 				# utility functions
│   ├── ukf.cpp 					# unscented kalman filter implementation
│   ├── ukf.h 					# header file for the unscented kalman filter
├── CMakeLists
├── Readme.md

```



