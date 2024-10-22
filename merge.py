import numpy as np
import cv2
def get_circle_count(img_path,img_path2, threshold=200, draw=True, name=None):
        # Denoising
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image2=cv2.imread(img_path2)
        denoisedImg = cv2.fastNlMeansDenoising(image)

        # Threshold (binary image)
        # thresh – threshold value.
        # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
        # type – thresholding type
        th, threshedImg = cv2.threshold(denoisedImg, threshold, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU) # src, thresh, maxval, type

        # Perform morphological transformations using an erosion and dilation as basic operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

        # Find and draw contours
        contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if draw:
            contoursImg = np.zeros_like(morphImg)
            contoursImg = np.repeat(contoursImg[:,:,np.newaxis],3,-1)
            for point in contours:
                x,y = point.squeeze().mean(0)
                if x==127.5 and y==127.5:
                    continue
                cv2.circle(image2, (int(x),int(y)), radius=2, thickness=-1, color=(0,255,0))
            # threshedImg = np.repeat(threshedImg[:,:,np.newaxis], 3,-1)
            # morphImg = np.repeat(morphImg[:,:,np.newaxis], 3,-1)
            # image = np.concatenate([contoursImg, threshedImg, morphImg], axis=1)
            cv2.imwrite('contour_merge.jpg', image2)
        return len(contours)
def main():
  print(get_circle_count("pred_density.png","out_data/shtech_A/part_1/test/1-12.jpg"))
