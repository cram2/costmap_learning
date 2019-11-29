#!/usr/bin/env python3

from costmap_learning.srv import GetCostmap, GetSymbolicLocation
from rospy import init_node, Service, spin

from learning import init_dataset, get_costmap, get_symbolic_location


def start_get_costmap_server():
    s = Service('get_costmap', GetCostmap, get_costmap)

def start_get_symbolic_location():
    s = Service('get_symbolic_location', GetSymbolicLocation, get_symbolic_location)
    
def start_services():
    init_node("learning_vr")
    start_get_costmap_server()
    start_get_symbolic_location()
    print("Ready")
    spin()


if __name__ == "__main__":
    init_dataset()
    start_services()
