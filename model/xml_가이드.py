from xml.etree.ElementTree import Element, SubElement, ElementTree
 
filename = '01_manual'
width = 500
height = 500
point1 = (300, 400)
point2 = (250, 350)
label = 'cross'
 
root = Element('annotation')
SubElement(root, 'folder').text = 'custom_images'
SubElement(root, 'filename').text = filename + '.gif'
SubElement(root, 'path').text = './object_detection/images' +  filename + '.gif'
source = SubElement(root, 'source')
SubElement(source, 'database').text = 'Unknown'
 
size = SubElement(root, 'size')
SubElement(size, 'width').text = str(width)
SubElement(size, 'height').text = str(height)
SubElement(size, 'depth').text = '1'
 
SubElement(root, 'segmented').text = '0'
 
obj = SubElement(root, 'object')
SubElement(obj, 'name').text = label
SubElement(obj, 'pose').text = 'Unspecified'
SubElement(obj, 'truncated').text = '0'
SubElement(obj, 'difficult').text = '0'
bbox = SubElement(obj, 'bndbox')
SubElement(bbox, 'xmin').text = str(point1[0])
SubElement(bbox, 'ymin').text = str(point1[1])
SubElement(bbox, 'xmax').text = str(point2[0])
SubElement(bbox, 'ymax').text = str(point2[1])                                                        
 
tree = ElementTree(root)
tree.write('./' + filename +'.xml')Colored by Color Scripter