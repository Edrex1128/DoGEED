

import cv2
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import argparse
from utils.functions import *
from YOLO_head import Detect
mpl.use('TkAgg')


ImgID = '1'  # Different numbers correspond to different images in the folder

# Initialize YOLOv5 parameters
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='' + ImgID + '.jpg', help='source')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()

# Initialize ACM parameters
delta1 = 0.1  # NDF iteration step size
iterNum1 = 10  # NDF iteration number
sigma1 = 0.5  # Gaussian kernel standard deviation 1
sigma2 = 4.5  # Gaussian kernel standard deviation 2
w = 10  # Gaussian kernel size
k = 5  # Mean filter kernel size
delta2 = 1  # Gradient descent flow iteration step size
iterNum2 = 100  # Gradient descent flow iteration number

# Input image
image0 = cv2.imread('' + ImgID + '.jpg')
image1 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
img = np.array(image1,dtype='float64')

# Initialize level set function
InitialLSF = np.ones([img.shape[0], img.shape[1]], img.dtype)
r1, r2, c1, c2 = Detect(opt)
InitialLSF[r1:r2, c1:c2] = -1

# Visualize initial contour
fig1 = plt.figure()
fig1.canvas.manager.set_window_title('Initial contour')
plt.contour(InitialLSF ,[0],linewidths = 2.0,linestyles = 'solid',colors='g')
plt.imshow(imutils.opencv2matplotlib(image0))

# Define DoG
H1 = gauss(w, sigma1)
H2 = gauss(w, sigma2)
DiffGauss = H1 - H2
Im = cv2.filter2D(img, -1, DiffGauss)
e = Im
eta = math.sqrt(9 * np.std(img))
s = np.std(img)

# NDF iteration
for i in range(iterNum1):
    EED(e, eta, delta1)

m = e.shape[0]
n = e.shape[1]
e_erf = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        e_erf[i, j] = math.erf((e[i, j] / eta) ** 2)

ex = 4 * np.sign(e) * e_erf

# Iterate initial contour
LSF = InitialLSF

tic = timer()
for i in range(1, iterNum2):
    LSF1 = LSF
    Drc = 0.5 * math.log(2) * 2 ** (-abs(LSF))
    LSF = LSF + delta2 * ex * Drc
    LSF = atan2(11 * LSF)
    LSF = cv2.blur(LSF, (k, k))
    if np.abs(LSF - LSF1).sum() < 0.001 * (LSF.shape[0] * LSF.shape[1]):
        break
toc = timer()
print()
print(f"Segmentation time: {toc - tic:.3f}s")

# Visualize final contour
fig2 = plt.figure()
fig2.canvas.manager.set_window_title('Final contour')
plt.contour(LSF,[0],linewidths = 2.0,linestyles = 'solid',colors='r')
plt.imshow(imutils.opencv2matplotlib(image0))
plt.show()

print("Segmentation complete.")

