import os
import cv2 as cv
import numpy as np

# Read Dataset
an = 1
if an == 1:
    File_Fold = os.listdir('./Occluded-Facial-Expression-Recognition-main/datasets/FED-RO/FED-RO_crop/')
    Images = []
    Target = []
    Prep = []
    for n in range(len(File_Fold)):
        out_fold = os.listdir('./Occluded-Facial-Expression-Recognition-main/datasets/FED-RO/FED-RO_crop/' + File_Fold[n])
        for m in range(len(out_fold)):
            Filename = './Occluded-Facial-Expression-Recognition-main/datasets/FED-RO/FED-RO_crop/' + File_Fold[n] + '/' + out_fold[m]
            img = cv.imread(Filename)
            image = cv.resize(img, (512, 512))
            Tar = n
            median = cv.medianBlur(image, 5)  # median Filtering
            alpha = 1.5  # Contrast control (1.0-3.0)
            beta = 0  # Brightness control (0-100)
            adjusted = cv.convertScaleAbs(median, alpha=alpha, beta=beta)  # Contrast Enhancement
            Images.append(np.uint8(image))
            Prep.append(adjusted)
            Target.append(Tar)
    uniq = np.unique(Target)

    Tars = np.zeros((len(Target), len(uniq)))
    for j in range(len(uniq)):
        Index = np.where(Target == uniq[j])
        Tars[Index[0], j] = 1

    np.save('Images.npy', Images)
    np.save('Preprocessed_Images.npy', Prep)
    # np.save('Target.npy', Tars)