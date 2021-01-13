import numpy as np
import torch, random, PIL
from torchvision import transforms
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torchio as tio
from torchio.data.io import nib_to_sitk, sitk_to_nib
from torchio import AFFINE, DATA
import matplotlib.pyplot as plt
"""
Crop the image
"""

class Crop_ROI():
    def __init__(self, maskName="ROI"):
        self.mask_name = maskName

    def threshold_based_crop(self, mask_sitk=None):
        inside_value = 0
        outside_value = 255
        mask = sitk.Cast(mask_sitk, sitk.sitkFloat32)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(sitk.OtsuThreshold(mask, inside_value, outside_value))
        bbox = label_shape_filter.GetBoundingBox(outside_value)
        return bbox

    def crop_img(self, imgs):
        mask_name = self.mask_name
        bbox = self.threshold_based_crop(imgs[mask_name].as_sitk())
        for img in imgs.get_images(intensity_only=False):
            img_sitk = img.as_sitk()
            cropped_img = sitk.RegionOfInterest(img_sitk, bbox[int(len(bbox)/2):], bbox[0:int(len(bbox)/2)])
            data, affine = sitk_to_nib(cropped_img)
            tensor = torch.from_numpy(data)
            img[DATA] = tensor
            img[AFFINE] = affine

        return imgs


class PseudoColor():
    def __init__(self, cmap="jet"):
        self.cmap = cmap



def main():
    subject_dic = {
    "image": tio.ScalarImage("/Users/chenziyan/Desktop/ToDoList/BraTS2013/HCG/brats_2013_pat0004_h/VSD.Brain.XX.O.MR_T1c.54532/VSD.Brain.XX.O.MR_T1c.54532.mha"),
    "label": tio.LabelMap("/Users/chenziyan/Desktop/ToDoList/BraTS2013/HCG/brats_2013_pat0004_h/VSD.Brain_3more.XX.O.OT.54535/VSD.Brain_3more.XX.O.OT.54535.mha")
    }
    subject = tio.Subject(subject_dic)
    Crop = Crop_ROI(maskName="label")
    res = Crop.crop_img(subject)
    a = res["image"].as_sitk()
    print(type(a))
    b = sitk.GetArrayFromImage(a)
    print(b.shape)
    plt.imshow(b[:, :, 0])


if __name__ == '__main__':
    main()






















