from costmap_learning.srv import GetCostmap
from rospy import init_node, Service, spin

from learning import init_dataset, lookup


def get_costmap_server():
    init_node('learning_get_costmap')
    s = Service('get_costmap', GetCostmap, lookup())
    spin()


if __name__ == "__main__":
    init_dataset()
    get_costmap_server()
