import cv2

root = r"/datav/Lidar_AI_Solution/CUDA-BEVFusion/bevfusion/data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604512404.jpg"
imag = cv2.imread(root)
print(imag.shape)

