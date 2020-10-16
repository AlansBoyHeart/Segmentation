#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F
import os
# from models import UNet
from dataloaders import transforms
from utils import utils
# from models import DeepLabV3Plus
from PIL import Image
from pylab import *
from models.PSPNet import PSPNet

device = "cuda" if torch.cuda.is_available() else "cpu"
#------------------------------------------------------------------------------
#   Argument parsing
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Arguments for the script")

parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='Use GPU acceleration')

parser.add_argument('--bg', type=str, default=None,
                    help='Path to the background image file')

parser.add_argument('--watch', action='store_true', default=False,
                    help='Indicate show result live')

parser.add_argument('--input_sz', type=int, default=1024,
                    help='Input size')
aaa = "/data/wuxiaopeng/workspace/project/Segmentation/ckpt/HumanSeg/0918_142639/model_best.pth"
# parser.add_argument('--checkpoint', type=str, default="/data/COMP/Human-Segmentation-PyTorch-master/workspace/checkpoints/HumanSeg/0913_074939/model_best.pth",
parser.add_argument('--checkpoint', type=str, default=aaa,

                    help='Path to the trained model file')

parser.add_argument('--video', type=str, default=None,
                    help='Path to the input video')

parser.add_argument('--output', type=str, default="../result",
                    help='Path to the output video')

args = parser.parse_args()


#------------------------------------------------------------------------------
#	Parameters
#------------------------------------------------------------------------------


# Background
if args.bg is not None:
	BACKGROUND = cv2.imread(args.bg)[...,::-1]
	BACKGROUND = cv2.resize(BACKGROUND, (W,H), interpolation=cv2.INTER_LINEAR)
	KERNEL_SZ = 25
	SIGMA = 0

# Alpha transperency
else:
	COLOR1 = [255, 0, 0]
	COLOR2 = [0, 0, 255]


#------------------------------------------------------------------------------
#	Create model and load weights
#------------------------------------------------------------------------------
model = PSPNet("resnet18", 2,"./ckpt/resnet18.pth")
# model = DeepLabV3Plus(
#     backbone="resnet18",
#     num_classes=2,
# 	pretrained_backbone=None
# )
if args.use_cuda:
	model = model.to(device)

bbb = "./model_best.pth"
# trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
trained_dict = torch.load(aaa, map_location="cpu")['state_dict']

model.load_state_dict(trained_dict, strict=False)
model.to(device)
model.eval()


#------------------------------------------------------------------------------
#   Predict frames
#------------------------------------------------------------------------------
i = 0
PATH = "/data/COMP/Train_Data/Train_Images"
PATH = "E:/dataset/Test_Images_2920"
PATH = "/data/wuxiaopeng/datasets/Test_Images_2920"
imgs = os.listdir(PATH)
#start = time()
for i in imgs:

	#pic_s = time()
	# Read frame from camera
	img = os.path.join(PATH,i)
	frame = cv2.imread(img)
	# image = cv2.transpose(frame[...,::-1])
	image = frame[...,::-1]
	h, w = image.shape[:2]


	# Predict mask
	X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=args.input_sz, pad_value=0)
	#preproc_time = time()
	with torch.no_grad():
		if args.use_cuda:
			mask = model(X.to(device))
			mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
			mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
			mask = F.softmax(mask, dim=1)
			mask = mask[0,1,...].cpu().numpy()
		else:
			mask = model(X)
			mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
			mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
			mask = F.softmax(mask, dim=1)
			mask = mask[0,1,...].numpy()

	# Draw result
	if args.bg is None:
		image_alpha = utils.draw_matting(image, mask)
		#ssimage_alpha = utils.draw_transperency(image, mask, COLOR1, COLOR2)
	else:
		image_alpha = utils.draw_fore_to_back(image, mask, BACKGROUND, kernel_sz=KERNEL_SZ, sigma=SIGMA)
	# draw_time = time()
	#
	# # Print runtime
	# read = read_cam_time-start_time
	# preproc = preproc_time-read_cam_time
	# pred = predict_time-preproc_time
	# draw = draw_time-predict_time
	# total = read + preproc + pred + draw
	# fps = 1 / total
	# print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; draw: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" %
	# 	(read, preproc, pred, draw, total, fps))
	out_img = image_alpha[..., ::-1]
#	gray = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
#	ret, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
	# Wait for interupt
	out_img = Image.fromarray(out_img).convert('L')
#	out_img = Image.fromarray(mask.astype('uint8'))
	thres=254
	table=[]
	for j in range(256):
		if j<thres:
			table.append(1)
		else:
			table.append(0)
	out_img = out_img.point(table,"1")
	out_img = Image.fromarray(uint8(out_img))
	name = i
#	name = name.split('.')[0]
#	name = name+"_.png"
	output_p = os.path.join(args.output,name)
	print(output_p)
	out_img.save(output_p)
#	cv2.imwrite(output_p,out_img)
	#pic_e = time()
#	print("单张时间:%.2f秒" % (pic_s - pic_e))
#	out.write(image_alpha[..., ::-1])
#end  = time()
#sssprint("时间:%.2f秒"%(end-start))