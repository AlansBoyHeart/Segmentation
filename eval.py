

import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import glob
import os
import cv2


def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n) #k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)#np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def compute_mIoU(gt_dir, pred_dir, num_classes =2):#计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and 
    """
    
    print('Num classes', num_classes)#打印一下类别数目
    
    hist = np.zeros((num_classes, num_classes))#hist初始化为全零，在这里的hist的形状是[20, 20]
 
    gt_imgs=glob.glob(gt_dir+"/*.*")
    pred_imgs=[]
    for gt_img in gt_imgs:
        a,b=os.path.split(gt_img)
        b=b.replace("gtFine_labelIds","leftImg8bit")
        pred_img=os.path.join(pred_dir,b)
        pred_imgs.append(pred_img)

    for ind in range(len(gt_imgs)):#读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]))#读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))#读取一张对应的标签，转化成numpy数组
      
        if len(label.flatten()) != len(pred.flatten()):#如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 100 == 0:#每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.4f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))
    
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):#逐类别输出一下mIoU值
        print('===>' +  ':\t' + str(round(mIoUs[ind_class] * 100, 4)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 4)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs

def test_iou():
    # aaa = "/data/wuxiaopeng/datasets/Train_Data/Train_Labels/"
    # bbb = "/data/wuxiaopeng/datasets/Train_Data/Train_Labels/"

    aaa = "E:\dataset\标签图像\label_2912"
    bbb = "E:\dataset\Test_Images_2920_zhongzhilvbo"

    compute_mIoU(aaa, bbb)

def same_pic__total_number():
    aaa = "E:\dataset\标签图像\label_2912"
    bbb = "E:\dataset\Test_Images_2920"
    data = os.listdir(aaa)
    for j in os.listdir(bbb):
        path1 = join(bbb,j)

        if j not in data:
            print(j)
            os.remove(path1)

def zhongzhilvbo():
    aaa = "E:\dataset\Test_Images_2920_zhongzhilvbo"
    for i in os.listdir(aaa):
        path1 = join(aaa,i)
        img = cv2.imread(path1,0)
        img = img * 255
        cv2.namedWindow("test1", cv2.WINDOW_FREERATIO)
        cv2.imshow("test1", img)
        img = cv2.medianBlur(img,5)
        # img = img*255
        cv2.namedWindow("test2",cv2.WINDOW_FREERATIO)
        cv2.imshow("test2",img)
        cv2.waitKey(0)

def zhongzhilvbo1():
    aaa = "E:\dataset\Test_Images_2920_zhongzhilvbo"
    for numi,i in enumerate(os.listdir(aaa)):
        path1 = join(aaa,i)
        img = cv2.imread(path1,0)
        img = cv2.medianBlur(img,9)
        # print(numi)
        cv2.imwrite(path1,img)

if __name__ == '__main__':
    # zhongzhilvbo1()
    test_iou()


