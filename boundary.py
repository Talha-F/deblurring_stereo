import numpy as np
import cv2
import os

"""
#crop
folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/sharp/right" # before padding imgs
for path in sorted(os.listdir(folder)):
    if path.endswith("g"):
        img = cv2.imread(os.path.join(folder, path))
        #img.shape
        #img.shape[0] - 370
        print(path)
        if (img.shape[0]-370)%2==0: #짝수
            crop_img = img[int((img.shape[0] - 370)/2):int((img.shape[0] - 370)/2)+370,img.shape[1]-1224:]
        else:
            crop_img = img[int((img.shape[0] - 370)/2):int((img.shape[0] - 370)/2)+370,img.shape[1]-1224:]
        cv2.imwrite("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/cropped/right/"+path,crop_img)
        #print(crop_img.shape)
"""


#padding (driving : 800*1762)
folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/cropped/right" # before padding imgs
size_dict = {}
a=0
for path in os.listdir(folder):
    #print(path)
    img = cv2.imread(os.path.join(folder, path))
    a+=1
    #print(os.path.join(folder, path))
    #print(type(img))
    #print(img.shape) 
    
    #if (img.shape in size_dict):
    #    size_dict[img.shape]+=1
    #else:
    #    size_dict[img.shape] = 1
    
    # kitti : 1224 * 370
    up_pad = int((512-int(img.shape[0]))/2)
    left_pad = int((2048-int(img.shape[1]))/2)
    padded_img = np.pad(img, ((up_pad,up_pad),(left_pad,left_pad),(0,0)), 'constant', constant_values=0)
    #padded_img = np.pad(img, ((71,71),(412,412),(0,0)), 'constant', constant_values=0)
    #padded_img = np.pad(img, ((112,112),(143,143),(0,0)), 'constant', constant_values=0)

    cv2.imwrite("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/padded/right/"+path,padded_img)
    print(a)



"""
folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti/blurred/kitti/padded_blurred_left"
for path in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, path))

    
    delete_list_row = []
    delete_list_column = []

    print(os.path.join(folder, path))

    for j in range(int(img.shape[0])): # 세로 줄 한 픽셀씩
        for k in range(int(img.shape[1])): # 가로 줄 한 픽셀씩
            #print(type(img[0][j]))
            
            if (np.array_equal(img[j][k],np.array([0,0,0]))):
                delete = True
            else:
                delete = False
                break

        if delete:
            delete_list_column.append(j)

    print(delete_list_column)
    print(len(delete_list_column))

    for j in range(int(img.shape[1])): # 세로 줄 한 픽셀씩
        for k in range(int(img.shape[0])): # 가로 줄 한 픽셀씩
            #print(type(img[0][j]))
            
            if (np.array_equal(img[k][j],np.array([0,0,0]))):
                delete = True
            else:
                delete = False
                break

        if delete:
            delete_list_row.append(j)

    print(delete_list_row)
    print(len(delete_list_row))

    
    black_cropped_img = np.delete(img, delete_list_row, axis=1)
    
    black_cropped_img = np.delete(black_cropped_img, delete_list_column, axis=0)
    cv2.imwrite("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/dataset_test/cropped_padding_test/"+"cropped_padded_"+path,black_cropped_img)

"""
"""
folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti/blurred/kitti/padded_blurred_left"

for path in os.listdir(folder):
    
    img = cv2.imread(os.path.join(folder, path))
    #print(img.shape)
    #print(int(img.shape[0])-370)
    #print(int(img.shape[1])-1224)
    crop_img = img[int(img.shape[0])-370:,int(img.shape[1])-1224:,:]
    
    cv2.imwrite("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti/blurred/kitti/blurred_final_left/"+path,crop_img)
"""
"""
#center crop
def center_crop(img, width, height):

    h, w, c = img.shape
    #512*2048
    #370*1224

    crop_width = width
    crop_height = height

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
        
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x,:]
    return crop_img

folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti_middle/blur/middle_padding_left"
for path in os.listdir(folder):
    
    img = cv2.imread(os.path.join(folder, path))
    #print(img.shape)
    #print(int(img.shape[0])-370)
    #print(int(img.shape[1])-1224)
    crop_img = center_crop(img,1224,370)
    
    cv2.imwrite("/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti_middle/blur/middle_padding_left_cropped/"+path,crop_img)
"""
"""
folder = "/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/blurred_image/kitti/sharp/kitti/image_left"
dic = {}
for path in os.listdir(folder):
    if(path.endswith("g")):
        img = cv2.imread(os.path.join(folder, path))
        if img.shape in dic:
            dic[img.shape]+=1
        else:
            dic[img.shape]=0
print(dic)
"""