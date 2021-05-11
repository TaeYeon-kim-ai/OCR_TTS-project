import xml.etree.cElementTree as ET
import os

def toxml(lines, save_filepath):

    def generate_xml(obj, cordinates_arr, width, height, depth, save_filepath):
        root_node = ET.Element("annotation")
        object_node = ET.SubElement(root_node, "object")
        ET.SubElement(object_node, "filename").text = obj + '.tif'
        
        root_node = ET.SubElement(root_node, "size")
        ET.SubElement(root_node, 'width').text = str(width)
        ET.SubElement(root_node, 'height').text = str(height)
        ET.SubElement(root_node, 'depth').text = str(depth)
        
        ET.SubElement(root_node, 'segmented').text = '0'

        #object
        boject = ET.SubElement(root_node, 'object')
        ET.SubElement(object, 'name').text = cordinates_arr[0]
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'occluded').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        cordinates_node = ET.SubElement(object, 'bnd')
        ET.SubElement(cordinates_node, 'xmin').text = cordinates_arr[1]
        ET.SubElement(cordinates_node, 'ymin').text = cordinates_arr[2]
        ET.SubElement(cordinates_node, 'xmax').text = cordinates_arr[3]
        ET.SubElement(cordinates_node, 'ymax').text = cordinates_arr[4]

        tree = ET.ElementTree(root_node)
        tree.write(save_filepath + filename + '.xml')
    #설정    
    if len(lines) != 2:
        print("Invalid content: {}".format(lines))
    obj = lines[0].split()
    cordinates = lines[1].split()

    if len(obj) == '' or len(cordinates.split()) != 4:
        print("Invalid line format: {}".format(lines))
    
    #image size
    IMAGE_WIDTH = 1820
    HIMAGE_EIGHT = 2570
    DEPTH = 3

    # start generate
    generate_xml(obj, cordinates, IMAGE_WIDTH, HIMAGE_EIGHT, DEPTH, save_filepath)

def entry(target_dir_path, save_dri_path):
    assert os.path.exists(target_dir_path), "Target directory is not exist: {}".format(target_dir_path)
    assert os.path.exists(save_dir_path), "Save directory is not exist: {}".format(target_dir_path)
    for filename in os.listdir(target_dir_path):
        file_full_path = os.path.join(target_dir_path, filename)
        filename_prefix, _ = os.path.splitext(filename)
        save_path = os.path.join(save_dir_path, "{}.xml".format(filename_prefix))
        try:
            with open(file_full_path, 'r', encoding='utf-8') as ff:
                toxml(ff.readlines(), save_path)
        except Exception as ex:
            print("Generate {0} failed, with error msg: {1}.".format(filename, ex.__str__()))

if __name__ == '__main__':
    target_dir_path = 'C:/final_project/model/test_data/test_box'
    save_dir_path = 'C:/final_project/model/test_data/test_xml'
    entry(target_dir_path, save_dir_path)


