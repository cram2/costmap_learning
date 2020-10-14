# Learning Position and Orientation Distributions based on GaussianMixtureModels (GMM)

This package allows to model with GMM a two dimensional position distribution and one dimensional orientation distribution
for different objects. The software architecture supports to use different kitchens, tables, humans and contexts
for learning different position and orientation distributions. The GMMs are learned from one CSV file, 
which must be saved in the [resource](../master/resource) folder and its name should be configured in the launch file. 
Moreover, the launch file allows to change the CSV file feature names, visualizations options during runtime and more.
This package already contains a data set for the iai_kitchen in iai_maps. Lastly, this packages supports ROS kinetic
and melodic.

## ROS Interface

The ROS-Interface has two services called `GetCostmap` and `GetSymbolicLocation`. The service `GetSymbolicLocation`
allows to return the symbolic storage or destination location of given object type. The service `GetCostmap` then
allows to return the parameters of the learned GMMs for a given object type and location. 

## Install

This python package needs atleast Python 3.6 and the following packages:

### Ubuntu 16.04

```
pip install --upgrade pip
sudo apt-get install python-catkin-pkg python3-pip
pip uninstall em
pip3 install pyyaml empy
python3.6 -m pip install numpy scipy pandas matplotlib seaborn sklearn 
```

### Ubuntu 18.04

```
pip install --upgrade pip
sudo apt-get install python-catkin-pkg python3-pip
pip uninstall em
pip install empy
pip3 install numpy scipy pandas matplotlib seaborn sklearn
```

After that this package can be built with `catkin_make` in your ROS workspace.

## Executing

To start the ROS service simply include the given launch file in your main launch file or start it with: 

```
roslaunch costmap_learning costmap_learning_with_params.launch 
```
