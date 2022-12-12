"""
import os
path = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/right'
print(len(os.listdir(path)))
file_name = os.listdir(path)

for name in file_name:
    src = os.path.join(path, name)
    dst = path+"/2012_"+name
    
    os.rename(src, dst)
"""
import os
print(len(os.listdir("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/final_dataset/final/left")))
print(len(os.listdir("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/sharp/image_right")))