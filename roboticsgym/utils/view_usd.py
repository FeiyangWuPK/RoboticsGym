# Import the necessary USD modules
from pxr import Usd


def print_usd_hierarchy(stage, node, indent=""):
    print(node.GetPath())

    for child in node.GetChildren():
        print_usd_hierarchy(stage, child, indent + "  ")


def view_usd_file(usd_file_path):
    # Create a USD stage to open the USD file
    stage = Usd.Stage.Open(usd_file_path)

    if not stage:
        print("Failed to open USD file")
        return

    # Start traversing the hierarchy from the root layer
    # stage.GetRootLayer()
    root_layer = stage.GetRootLayer()
    root_prim = stage.GetPseudoRoot()

    print("USD Hierarchy:")
    print_usd_hierarchy(stage, root_prim)


if __name__ == "__main__":
    usd_file_path = "/home/feiyang/Research/Robotics/OmniIsaacGymEnvs-Bipedal/omniisaacgymenvs/assets/USD/digit/digit_float.usd"  # Replace with the path to your USD file
    view_usd_file(usd_file_path)
