from xml.etree import ElementTree as ET


def replace_dashes_in_attributes(xml_file):
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Define attributes to be modified
        attributes_to_modify = ["name", "joint", "class", "mesh", "target"]

        # Iterate over all elements and their attributes
        for element in root.iter():
            for attr in attributes_to_modify:
                if attr in element.attrib:
                    # Replace dashes with underscores
                    element.attrib[attr] = element.attrib[attr].replace("-", "_")

        # Convert the tree to a string
        modified_xml = ET.tostring(root, encoding="unicode")
        return modified_xml
    except Exception as e:
        return f"Error processing the file: {e}"


def save_xml(tree, output_file):
    """
    Save an ET object to an XML file.

    :param tree: An ElementTree object representing the XML tree.
    :param output_file: Path to the output file where the XML should be saved.
    """
    try:
        # Write the tree to the file
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"XML file saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")


def save_xml_string(xml_string, output_file):
    """
    Save a string representation of an XML to a file.

    :param xml_string: String representation of the XML.
    :param output_file: Path to the output file where the XML should be saved.
    """
    try:
        # Open the file in write mode
        with open(output_file, "w", encoding="utf-8") as file:
            # Write the XML string to the file
            file.write(xml_string)
        print(f"XML file saved successfully to {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")


# Replace 'your_xml_file.xml' with the path to your MuJoCo XML file.
modified_xml = replace_dashes_in_attributes(
    "C:\\Users\\fwupk\\OneDrive\\Documents\\Research\\Robot Learning\\RoboticsGym\\roboticsgym\\envs\\xml\\mj_cassie.xml"
)
# print(modified_xml)
save_xml_string(modified_xml, "mj_cassie_modified.xml")
