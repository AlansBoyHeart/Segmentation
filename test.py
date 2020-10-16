import os,cv2, time
import numpy as np
from multiprocessing import Manager,Pool,Process
def test1():
	aaa = "E:\dataset/result"

	for i  in os.listdir(aaa):
		path = os.path.join(aaa,i)
		img = cv2.imread(path)
		for j in img:
			for k in j:
				for z in k:
					if z == 1:
						print("###############")
					else:
						print(z)

		time.sleep(3)

def test2():
	path = "D:/Users\Focusmedia\Desktop/111.jpg"
	image = cv2.imread(path)[..., ::-1]
	print(image.shape)
	aa = np.array([[12,14],[13,16]])
	bb = aa[...,::1]   #省略号代表前面的所有维度,::-1,表示start：end：step，意思是从头到尾逆序排序，如果是：：1表示不发生变化。
	print(bb)


# def f(d,l):
#     d['a'] = 888
#     l[1] = 99999
#     l.reverse()


test2()