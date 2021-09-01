import os
import shutil

dir_all = "pointer"

os.system("sudo chmod -R 777 " + dir_all)

out_data = os.path.join("/data/inspection/images", dir_all)
dir_list = os.listdir(dir_all)
dir_list.sort()
bef = 0
for dir_ in dir_list:
    dir_int = int("".join(dir_.split("-")))
    if dir_int - bef > 10000:
        cp_file = os.path.join(dir_all, dir_, "img_tag.jpg")
        mv_file = os.path.join(out_data, dir_ + ".jpg")
        shutil.copy(cp_file, mv_file)
        bef = dir_int
        print(dir_)

os.system("rm -r "+dir_all)
