# Learning Position and Orientation Distributions based on GaussianMixtureModels (GMM)

This package allows to learn a two dimensional position distribution and one dimensional orientation distributions
for different objects. The software architecture supports to use different kitchens, tables, humans and context
for learning different postion and orientation distributions. The GMMs are learned from one [CSV file](../master/resource), which is 
exemplary given for the iai_kitchen in iai_maps. Moreover, a launch file is given, which allows to change the CSV
file feature names, visualizations options and more.

## ROS Interface

The ROS-Interface has two services called `GetCostmap` and `GetSymbolicLocation`. The service `GetSymbolicLocation`
allows to return the symbolic storage or destination location of given object type. The service `GetCostmap` then
allows to return the parameters of the learned GMMs for a given object type and location. 

## Install

This python package supports Python 2 and 3 and needs some packages, which can be installed with the following command:

```
pip install numpy scipy pandas matplotlib seaborn sklearn
```

Moreover, the ros dependencies can be installed with:

```
rosdep install costmap_learning
```

## Executing

To start the ROS service simply include the given launch file in your main launch file or start it with: 

```
roslaunch costmap_learning costmap_learning_with_params.launch 
```

