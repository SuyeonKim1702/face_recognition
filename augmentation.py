import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
import glob



def write_images(j,images):
    for i in range(0,len(images)): 
        cv2.imwrite("C:\\Users\\td170\\Downloads\\dogcat\\dog\\"+"dog_"+str(j)+"_"+str(i)+".png", images[i][0]) 
        #이미지 저장할 경로 설정을 여기서 한다. print("image saving complete")


def augmentations1(images):  
    seq1 = iaa.PerspectiveTransform(scale=(0.01, 0.15)) # perspective trans
    seq2 = iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25) #elastictrans
    seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5) # 점모양 노이즈
    seq4 = iaa.PiecewiseAffine(scale=(0.01, 0.03)) #piecewiseaffine
    seq5 = iaa.ShearX((-20, 20))
    seq6 = iaa.Affine(scale=(0.5, 1.5))#scale
    seq7 = iaa.CropAndPad(percent=(-0.17, 0.17))
    seq8 = iaa.pillike.FilterSharpen()
    
    print("image augmentation beginning") 
    img1=seq1.augment_images(images)  
    img2=seq2.augment_images(images)  
    img3=seq3.augment_images(images) 
    img4=seq4.augment_images(images) 
    img5=seq5.augment_images(images) 
    img6=seq6.augment_images(images) 
    img7=seq7.augment_images(images) 
    img8=seq8.augment_images(images)
    
    print("proceed to next augmentations") 
    list = [img1, img2, img3, img4, img5, img6, img7, img8]
    return list




images = glob.glob(r'C:\\Users\\td170\\Downloads\\dogcat\\newdog\\*')
print(images)


j=1
for image_file in images:
 # 이미지 파일
     
    image = cv2.imread(image_file)
    a=[]
    a.append(image)
    photos_augmented1234 = augmentations1(a)
    print(photos_augmented1234[0])
    write_images(j, photos_augmented1234)
    j=j+1


    
