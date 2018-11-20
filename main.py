import numpy as np
from scipy import signal
import os
import cv2
import matplotlib.pyplot as plt

def get_optical_flow(I1g, I2g, window_size, tau=1e-2):
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
	w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
	I1g = I1g / 255. # normalize pixels
	I2g = I2g / 255. # normalize pixels
	# Implement Lucas Kanade
	# for each point, calculate I_x, I_y, I_t
	mode = 'same'
	fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
	fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
	ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + \
		 signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
	u = np.zeros(I1g.shape)
	v = np.zeros(I1g.shape)
	# within window window_size * window_size
	for i in range(w, I1g.shape[0]-w):
		for j in range(w, I1g.shape[1]-w):
			Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
			Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
			It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
			b = np.reshape(It, (It.shape[0],1)) # get b here
			A = np.vstack((Ix, Iy)).T # get A here
			# if threshold τ is larger than the smallest eigenvalue of A'A:
			if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
				nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
				u[i,j]=nu[0]
				v[i,j]=nu[1]
	return (u,v)

def get_optical_flow_triangle(oldImage, newImage, depth=6):
	oldI = oldImage
	newI = newImage
	finalU = np.zeros(newImage.shape)
	finalV = np.zeros(newImage.shape)
	for i in range(depth):
		oldI = cv2.resize(oldI, None, fx=0.5, fy=0.5)
		newI = cv2.resize(newI, None, fx=0.5, fy=0.5)
		u, v = get_optical_flow(oldI,newI,5)
		finalU += cv2.resize(u, (oldImage.shape[1],oldImage.shape[0]))
		finalV += cv2.resize(v, (oldImage.shape[1],oldImage.shape[0]))
	return (finalU, finalV)
	
def load_video(subDirectory, fileName):
	full_path = os.path.join(subDirectory, fileName)
	return cv2.VideoCapture(full_path)
	
def plot(oldImage, newImage, flow):
	plt.subplot(1,5,1), plt.xticks([]), plt.yticks([]), plt.title("old"), plt.imshow(oldImage, cmap='gray')
	plt.subplot(1,5,2), plt.xticks([]), plt.yticks([]), plt.title("new"), plt.imshow(newImage, cmap='gray')
	plt.subplot(1,5,3), plt.xticks([]), plt.yticks([]), plt.title("u"), plt.imshow(flow[0], cmap='gray')
	plt.subplot(1,5,4), plt.xticks([]), plt.yticks([]), plt.title("v"), plt.imshow(flow[1], cmap='gray')
	temp = np.sqrt(np.add(np.square(flow[0]),np.square(flow[1])))
	plt.subplot(1,5,5), plt.xticks([]), plt.yticks([]), plt.title("sqrt(u^2+v^2)"), plt.imshow(temp, cmap='gray')
	plt.show()
	
def go_through_flow(video, skipframes=2, startframe=0):
	if (video.isOpened()):
		for i in range(startframe):
			ret,frame = video.read()
		ret,frame = video.read()
		old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_counter = 0
		while(video.isOpened()):
			for i in range(frame_counter):
				ret,frame = video.read()
			ret, frame = video.read()
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			plot(old_frame,new_frame,get_optical_flow_triangle(old_frame,new_frame,5))
			old_frame = new_frame
			
video = load_video("data","chain_link_fence.mp4")
go_through_flow(video, skipframes=1)

