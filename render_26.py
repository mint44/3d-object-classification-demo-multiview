'''

Render 26 imgs from 26 different camera viewpoints surrounding the object.
using trimesh library, xvbf's required if running on a headless linux.

'''
import sys
import numpy as np
import trimesh
import os   
from fnmatch import fnmatch

IMG_RESOLUTION = np.array([224,224])

def render_view(obj_path, img_out_dir, sn,en):
    print 'rendering ', obj_path 
    if not os.path.exists(img_out_dir):
         os.makedirs(img_out_dir)
            
    # load a mesh
    mesh  = trimesh.load(obj_path)
    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()
    
    camera_orig, _geometry = scene.graph['camera']
    rotate_1 = trimesh.transformations.rotation_matrix(np.radians(45.0), [1,0,0], 
                                                     scene.centroid)
    camera_1 = np.dot(camera_orig,rotate_1)

    camera_2 = camera_orig

    rotate_3 = trimesh.transformations.rotation_matrix(np.radians(45.0), [-1,0,0], 
                                                    scene.centroid)
    camera_3 = np.dot(camera_orig,rotate_3)

    rotate_4 = trimesh.transformations.rotation_matrix(np.radians(90.0), [1,0,0], 
                                                    scene.centroid)
    camera_4 = np.dot(camera_orig,rotate_4)

    rotate_5 = trimesh.transformations.rotation_matrix(np.radians(90.0), [-1,0,0], 
                                                    scene.centroid)
    camera_5 = np.dot(camera_orig,rotate_5)

    rotate = trimesh.transformations.rotation_matrix(np.radians(45.0), [0,1,0], 
                                                     scene.centroid)


    camera_view_list = list()

    for i in range(8):
        camera_view_list.append(camera_1)
        camera_1 = np.dot(camera_1,rotate)
    for i in range(8):
        camera_view_list.append(camera_2)
        camera_2 = np.dot(camera_2,rotate)
    for i in range(8):
        camera_view_list.append(camera_3)
        camera_3 = np.dot(camera_3,rotate)
    camera_view_list.append(camera_4)
    camera_view_list.append(camera_5)
    for i in range(sn,en):
        trimesh.constants.log.info('Saving image %d', i)
        
        # rotate the camera view transform
        # camera_old, _geometry = scene.graph['camera']
        # print(camera_old)
        # camera_new = np.dot(camera_old, rotate)

        # apply the new transform
        scene.graph['camera'] = camera_view_list[i]
    
        # increment the file name
        file_name = str(i) + '.png'
        file_path = os.path.join(img_out_dir,file_name)
        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        if not '-nw' in sys.argv:
            # save a render of the object as a png
            with open(file_path, "wb") as file:
                file.write(scene.save_image(resolution=IMG_RESOLUTION, visible=False, flags={'cull':False}))


if __name__ == '__main__':


    ## get input lists
    # input_lst = []
    # for path, subdirs, files in os.walk(DATA_INPUT_DIR):
    #     for name in files:
    #         if fnmatch(name, '*.ply'):
    #             input_lst.append(name)

    # for obj_fn in input_lst:
    #     obj_path = os.path.join(DATA_INPUT_DIR,obj_fn)
    #     obj_out_dir = os.path.join(DATA_OUTPUT_DIR, obj_fn[:-4])
    #     if not os.path.exists(obj_out_dir):
    #         os.makedirs(obj_out_dir)
    #     render_view(obj_path, obj_out_dir)
    obj_out_dir = sys.argv[2]
    sn = int(sys.argv[3]) 
    en = int(sys.argv[4]) 
    print(sys.argv[1])
    
    render_view(sys.argv[1],obj_out_dir,sn,en)
