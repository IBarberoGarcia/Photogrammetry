import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as Rot
from PIL import Image

def read_ply(ply_file):
    '''Gets an ASCII ply file and returns a dictionary with x,y,z,nx,ny,nz,red,green,blue, faces (the ones that are presented
    in the file. If some value is missed (eg. no normals) the dict will not have that value.'''
    properties=[]#List of property names
    with open(ply_file, 'r') as f:
        lines = f.readlines()
    j=0
    faces_num=0
    for line in lines:
        if line.startswith('element vertex'):
            verts_num = int(line.split(' ')[-1])
        elif line.startswith('element face'):
            faces_num = int(line.split(' ')[-1])
        elif line.startswith('property'):
            properties.append(line.split(' ')[-1].strip('\n'))
        elif line.startswith('end_header'):
            start_line=j+1
            break
        j+=1
    
    ply_dict={}    
    verts_lines = lines[start_line:start_line + verts_num]
    
    verts = np.array([list(map(float, l.strip().split(' '))) for l in verts_lines])
    if faces_num>0:
        faces_lines = lines[start_line + verts_num:]
        faces = np.array([list(map(int, l.strip().split(' '))) for l in faces_lines])[:,1:]
        ply_dict['faces'] = faces
        
    i=0
    while i<len(properties):
        attr=properties[i]
        if attr == "x":
            ply_dict['x'] = verts.transpose()[i]
        elif attr == "y":
            ply_dict['y'] = verts.transpose()[i]
        elif attr == "z":
            ply_dict['z'] = verts.transpose()[i]
        elif attr == "nx":
            ply_dict['nx'] = verts.transpose()[i]
        elif attr == "ny":
            ply_dict['ny'] = verts.transpose()[i]
        elif attr == "nz":
            ply_dict['nz'] = verts.transpose()[i]
        elif attr == "red":
            ply_dict['red'] = [int(num) for num in verts.transpose()[i]]
        elif attr == "green":
            ply_dict['green'] = [int(num) for num in verts.transpose()[i]]
        if attr == "blue":
            ply_dict['blue'] = [int(num) for num in verts.transpose()[i]]
        i+=1
    return ply_dict


def cloud2pano(ply_file, EOP, cam, output_image, background_image=False, color=False):
    '''Generates a panoramic image from a point cloud, given origin and rotations,
    -ply_file: ASCII point cloud
    -EOP: List of external orientation parameters for pano, x0,y0,z0,omega,phi,kappa. Meters and radians.
    -cam: camera parameters file, contains pixel size of the image
        x_size, y_size
    -output_image: name and path for pano image to be created
    -background image: optional, path and name or false
    -color: use cloud color, if false all points are red'''
    
    x0 = EOP[0]
    y0 = EOP[1]
    z0 = EOP[2]
    ome = EOP[3]
    phi = EOP[4]
    az = -EOP[5]
    
    cam = open(cam,'r')
    cam = cam.readlines()
    x_size=int(cam[0].split()[0])
    y_size=int(cam[0].split()[1])
    

    ply_dict = read_ply(ply_file)
    x_list=ply_dict['x']
    y_list=ply_dict['y']
    z_list=ply_dict['z']
    red=ply_dict['red']
    green=ply_dict['green']
    blue=ply_dict['blue']
    i=0
    
    if background_image==False:
        image = np.zeros(shape=[y_size, x_size, 3], dtype=np.uint8)
    else:
        image = cv2.imread(background_image)
    
    if az!=0 or phi!=0 or ome!=0:
        x_list=x_list-x0
        y_list=y_list-y0
        r = Rot.from_euler('xyz', [ome, phi, az], degrees=False)
        pts = r.apply(np.transpose([ x_list, y_list, z_list]))
        x_list = np.transpose(pts)[0]
        y_list=np.transpose(pts)[1]
        x_list=x_list+x0
        y_list=y_list+y0
        
    for x in x_list:
        y=y_list[i]
        z=z_list[i]
        d_hor=((x-x0)**2+(y-y0)**2)**0.5 
        r=x_size/(2*math.pi)
        x_img = -int((np.arctan2((y-y0),(x-x0)))*r)
        y_img = y_size-int((0.5*y_size)+(int((z-z0) * (r/d_hor))))
        try:
            if color:
                image[y_img][x_img]=[red[i], green[i], blue[i]]
            else:
                image[y_img][x_img]=[0, 0, 255]
        except:
            pass
        i+=1
    cv2.imwrite(output_image,image)


EOP = [58.35850000148184, 5.254394235317385, 0.6713243125150195, -0.002883227519013609, -0.0071506130407059405, 2.998822426375224]
ply_file = "cloud_simple.ply"
cam = 'camera.txt'
output_image= 'panocloud.jpg'
cloud2pano(ply_file, EOP, cam, output_image, background_image=False, color=False)