from utils.vis_utils import show_obj_skel
from utils.rig_parser import Info

if __name__ == '__main__':
    mesh_file_name = "data/obj/17872.obj"
    rig_file_name = "data/rig_info/17872.txt"
    rig_info = Info(rig_file_name)
    show_obj_skel(mesh_file_name, rig_info.root)