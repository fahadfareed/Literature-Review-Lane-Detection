import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle 
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks_cwt
import scipy.misc

from moviepy.editor import VideoFileClip
from IPython.display import HTML
pi = 3.14159

import utils

def detect(fname):

    image = mpimg.imread(fname+'.jpeg')
    height, width = image.shape[:2]
    image = cv2.resize(image, (1280, 720))[:,:,:3]
    image_original = image
    kernel_size = 5
    img_size = np.shape(image)

    ht_window = np.uint(img_size[0]/1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1]/2)
    ctl_window = c_window - .36*np.uint(img_size[1]/2)
    ctr_window = c_window + .36*np.uint(img_size[1]/2)
    cbl_window = c_window - 0.9*np.uint(img_size[1]/2)
    cbr_window = c_window + 0.9*np.uint(img_size[1]/2)

    src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])
    dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],
                      [img_size[1],0],[0,0]])

    warped,M_warp,Minv_warp = utils.warp_image(image, src, dst, (img_size[1] , img_size[0]))
    image_HSV = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)

    yellow_hsv_low  = np.array([ 0,  100,  100])
    yellow_hsv_high = np.array([ 80, 255, 255])


    res_mask = utils.color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    res = utils.apply_color_mask(image_HSV,warped,yellow_hsv_low,yellow_hsv_high)
    image_HSV = cv2.cvtColor(warped,cv2.COLOR_RGB2HSV)

    white_hsv_low  = np.array([ 0,   0,   160])
    white_hsv_high = np.array([ 255,  80, 255])

    res1 = utils.apply_color_mask(image_HSV,warped,white_hsv_low,white_hsv_high)

    mask_yellow = utils.color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = utils.color_mask(image_HSV,white_hsv_low,white_hsv_high)
    mask_lane = cv2.bitwise_or(mask_yellow,mask_white)

    image = utils.gaussian_blur(warped, kernel=5)
    image_HLS = cv2.cvtColor(warped,cv2.COLOR_RGB2HLS)

    img_gs = image_HLS[:,:,1]
    sobel_c = utils.sobel_combined(img_gs)
    img_abs_x = utils.abs_sobel_thresh(img_gs,'x',5,(50,225))
    img_abs_y = utils.abs_sobel_thresh(img_gs,'y',5,(50,225))

    wraped2 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

    img_gs = image_HLS[:,:,2]
    sobel_c = utils.sobel_combined(img_gs)
    img_abs_x = utils.abs_sobel_thresh(img_gs,'x',5,(50,255))
    img_abs_y = utils.abs_sobel_thresh(img_gs,'y',5,(50,255))

    wraped3 = np.copy(cv2.bitwise_or(img_abs_x,img_abs_y))

    image_cmb = cv2.bitwise_or(wraped2,wraped3)
    image_cmb = utils.gaussian_blur(image_cmb,3)
    image_cmb = cv2.bitwise_or(wraped2,wraped3)

    image_cmb1 = np.zeros_like(image_cmb)
    image_cmb1[(mask_lane>=.5)|(image_cmb>=.5)]=1

    mov_filtsize = img_size[1]/50.
    mean_lane = np.mean(image_cmb1[img_size[0]/2:,:],axis=0)
    indexes = find_peaks_cwt(mean_lane,[100], max_distances=[800])
    
    window_size=50
    val_ind = np.array([mean_lane[indexes[i]] for i in range(len(indexes)) ])
    ind_sorted = np.argsort(-val_ind)

    ind_peakR = indexes[ind_sorted[0]]
    ind_peakL = indexes[ind_sorted[1]]
    
    if ind_peakR<ind_peakL:
        ind_temp = ind_peakR
        ind_peakR = ind_peakL
        ind_peakL = ind_temp

    n_vals = 8

    ind_min_L = ind_peakL-50
    ind_max_L = ind_peakL+50

    ind_min_R = ind_peakR-50
    ind_max_R = ind_peakR+50

    mask_L_poly = np.zeros_like(image_cmb1)
    mask_R_poly = np.zeros_like(image_cmb1)
    ind_peakR_prev = ind_peakR
    ind_peakL_prev = ind_peakL

    for i in range(8):
        img_y1 = img_size[0]-img_size[0]*i/8
        img_y2 = img_size[0]-img_size[0]*(i+1)/8
        
        mean_lane_y = np.mean(image_cmb1[img_y2:img_y1,:],axis=0)
        indexes = find_peaks_cwt(mean_lane_y,[100], max_distances=[800])
            
        if len(indexes)>1.5:
            val_ind = np.array([mean_lane[indexes[i]] for i in range(len(indexes)) ])
            ind_sorted = np.argsort(-val_ind)

            ind_peakR = indexes[ind_sorted[0]]
            ind_peakL = indexes[ind_sorted[1]]
            
            if ind_peakR<ind_peakL:
                ind_temp = ind_peakR
                ind_peakR = ind_peakL
                ind_peakL = ind_temp
                
        else:        
            if len(indexes)==1:
                if np.abs(indexes[0]-ind_peakR_prev)<np.abs(indexes[0]-ind_peakL_prev):
                    ind_peakR = indexes[0]
                    ind_peakL = ind_peakL_prev
                else:
                    ind_peakL = indexes[0]
                    ind_peakR = ind_peakR_prev
            else:
                ind_peakL = ind_peakL_prev
                ind_peakR = ind_peakR_prev
                           
        if np.abs(ind_peakL-ind_peakL_prev)>=100:
            ind_peakL = ind_peakL_prev
        if np.abs(ind_peakR-ind_peakR_prev)>=100:
            ind_peakR = ind_peakR_prev
                
        mask_L_poly[img_y2:img_y1,ind_peakL-window_size:ind_peakL+window_size] = 1.     
        mask_R_poly[img_y2:img_y1,ind_peakR-window_size:ind_peakR+window_size] = 1. 
       
        ind_peakL_prev = ind_peakL
        ind_peakR_prev = ind_peakR

    mask_L_poly,mask_R_poly = utils.get_initial_mask(image_cmb1,50, mean_lane)
    mask_L = mask_L_poly
    img_L = np.copy(image_cmb1)
    img_L = cv2.bitwise_and(img_L,img_L,mask=mask_L_poly)

    mask_R = mask_R_poly
    img_R = np.copy(image_cmb1)
    img_R = cv2.bitwise_and(img_R,img_R,mask=mask_R_poly)

    vals = np.argwhere(img_L>.5)
    all_x = vals.T[0]
    all_y =vals.T[1]

    left_fit = np.polyfit(all_x, all_y, 2)
    left_y = np.arange(11)*img_size[0]/10
    left_fitx = left_fit[0]*left_y**2 + left_fit[1]*left_y + left_fit[2]

    vals = np.argwhere(img_R>.5)

    all_x = vals.T[0]
    all_y =vals.T[1]

    right_fit = np.polyfit(all_x, all_y, 2)
    right_y = np.arange(11)*img_size[0]/10
    right_fitx = right_fit[0]*right_y**2 + right_fit[1]*right_y + right_fit[2]        

    window_sz = 20
    mask_L_poly = np.zeros_like(image_cmb1)
    mask_R_poly = np.zeros_like(image_cmb1)

    left_pts = []
    right_pts = []

    pt_y_all = []

    for i in range(8):
        img_y1 = img_size[0]-img_size[0]*i/8
        img_y2 = img_size[0]-img_size[0]*(i+1)/8
        
        pt_y = (img_y1+img_y2)/2
        pt_y_all.append(pt_y)
        left_pt = np.round(left_fit[0]*pt_y**2 + left_fit[1]*pt_y + left_fit[2])
        right_pt = np.round(right_fit[0]*pt_y**2 + right_fit[1]*pt_y + right_fit[2])
        
        right_pts.append(right_fit[0]*pt_y**2 + right_fit[1]*pt_y + right_fit[2])
        left_pts.append(left_fit[0]*pt_y**2 + left_fit[1]*pt_y + left_fit[2])
        
    warp_zero = np.zeros_like(image_cmb1).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 255))

    col_L = (255,255,0)
    col_R = (255,255,255)    

    utils.draw_pw_lines(color_warp,np.int_(pts_left),col_L)
    utils.draw_pw_lines(color_warp,np.int_(pts_right),col_R)

    newwarp = cv2.warpPerspective(color_warp, Minv_warp, (image.shape[1], image.shape[0])) 
    result = cv2.addWeighted(image_original, 1, newwarp, 0.5, 0)
    
    grid = []
    coordinates = []
    
    a = [[left_fitx[i], i*72] for i in range(0, 11)]
    b = [[right_fitx[i], i*72] for i in range(0, 11)]
    c = np.concatenate([a, b])

    c = np.array([c], dtype='float32')

    coordinates = cv2.perspectiveTransform(c, Minv_warp)[0]

    return coordinates, result