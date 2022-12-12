import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
from generate_PSF import PSF
from generate_trajectory import Trajectory
from datetime import datetime

now = datetime.now()

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



class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = cv2.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images yet.')
            #elif self.shape[0] != self.shape[1]:
            #    raise Exception('We support only square images yet.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[0]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[0], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def blur_image(self, save=True, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                
                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #blured = np.zeros(self.shape)
                blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_32F)
                
                blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                #blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            #print(tmp)
            #print(tmp.shape)
            cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)
            #print("start")
            #print(now.time())
            blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            #print("0 done")
            #print(now.time())
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            #print("1 done")
            #print(now.time())
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
            #print("2 done")
            #print(now.time())
            blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #print("normalize done")
            #print(now.time())
            
            #blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            result.append(np.abs(blured))
        self.result = result
        if show or save:
            self.__plot_canvas(show, save)

    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                #cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                
                
                    
                #img = cv2.imread(os.path.join(folder, path))
                #print(img.shape)
                #print(int(img.shape[0])-370)

                
                #print(int(img.shape[1])-1224)

                #
                
                crop_img = center_crop(self.result[0] * 255,1224,370)
                
                cv2.imwrite(self.path_to_save+"/"+self.image_path.split('/')[-1],crop_img)
                

                
                
                
                print('saved')
            elif show:
                plt.show()


if __name__ == '__main__':
    # 
    
    
    folder_left = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/padded/left'
    folder_to_save_left = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/final_dataset/15_3/left'
    a = 1
    

    
    folder_right = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/KITTI/padded/right'
    folder_to_save_right = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/final_dataset/15_3/right'
    

    params = [0.0]
    #params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    trajectory = Trajectory(canvas=64, max_len=15, expl=np.random.choice(params),path_to_save="/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/final_dataset/15_3/trajectory.png").fit()
    #part=np.random.choice([1,2,3])
    part = 3
    psf = PSF(canvas=64, trajectory=trajectory, path_to_save="/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/FINAL/final_dataset/15_3/psf.png").fit()
    for path in os.listdir(folder_left):
        print(a)
        a+=1
        BlurImage(os.path.join(folder_left, path), PSFs=psf,
                  path__to_save=folder_to_save_left, part=part).\
            blur_image(save=True)
        
        BlurImage(os.path.join(folder_right, path), PSFs=psf,
                  path__to_save=folder_to_save_right, part=part).\
            blur_image(save=True)
        
    
    """
    
    folder_left = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/left'
    folder_to_save_left = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/15_3/left'
    a = 1
    folder_right = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/right'
    folder_to_save_right = '/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/15_3/right'

    params = [0.0]
    #params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    trajectory = Trajectory(canvas=64, max_len=15, expl=np.random.choice(params),path_to_save="/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/15_3/trajectory.png").fit()
    #part=np.random.choice([1,2,3])
    part = 3
    psf = PSF(canvas=64, trajectory=trajectory, path_to_save="/Users/yujin_kim/Desktop/IDL_datageneration/DeblurGAN/final_dataset/new/15_3/psf.png").fit()
    for path in os.listdir(folder_left):
        print(a)
        a+=1
        BlurImage(os.path.join(folder_left, path), PSFs=psf,
                  path__to_save=folder_to_save_left, part=part).\
            blur_image(save=True)

        BlurImage(os.path.join(folder_right, path), PSFs=psf,
                  path__to_save=folder_to_save_right, part=part).\
            blur_image(save=True)
        
    
    """