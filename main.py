import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

	
def get_optical_flow(Old_Image, New_Image, window_size=5, debug_level=0, mask=None):
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	
	Old_Image = Old_Image / 255. # normalize pixels
	New_Image = New_Image / 255. # normalize pixels
	
	Ix = cv2.filter2D(Old_Image,-1,kernel_x)
	Iy = cv2.filter2D(Old_Image,-1,kernel_y)
	It = New_Image - Old_Image
	
	if debug_level>0:
		plot_image(Ix, title="partial x derivative of old image")
		plot_image(Iy, title="partial y derivative of old image")
		plot_image(It, title="partial time derivative of images")
	
	u = np.zeros(Old_Image.shape)
	v = np.zeros(Old_Image.shape)
	
	w = int(window_size/2)

	for i in range(w, Old_Image.shape[0]-w):
		for j in range(w, Old_Image.shape[1]-w):
			if mask is None or mask[i][j][0] == 0.0:
				tempIx = Ix[i-w:i+w+1, j-w:j+w+1].flatten()
				tempIy = Iy[i-w:i+w+1, j-w:j+w+1].flatten()
				tempIt = It[i-w:i+w+1, j-w:j+w+1].flatten()
				b = np.reshape(tempIt, (tempIt.shape[0],1)) # get b here
				A = np.vstack((tempIx, tempIy)).T # get A here
			
				eigns = np.linalg.eigvals(np.matmul(A.T, A))
				eign1 = eigns[0]
				eign2 = eigns[1]
				if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= 1e-2:
					if True:
						x = np.linalg.inv(A.T @ A) @ A.T @ b
						u[i,j] = x[0]
						v[i,j] = x[1]
	return (u,v)

def get_optical_flow_triangle(oldImage, newImage, window_size=5, depth=6, debug_level=0, mask=None):
	oldI = oldImage
	newI = newImage
	finalU,finalV = get_optical_flow(oldI,newI,window_size=window_size,debug_level=debug_level-1,mask=mask)
	for i in range(depth):
		oldI = cv2.pyrDown(oldI)
		newI = cv2.pyrDown(newI)
		if mask is not None:
			mask = np.floor(cv2.resize(np.array(mask,dtype=float), (int(oldI.shape[1]),int(oldI.shape[0])), interpolation = cv2.INTER_LINEAR))
		u, v = get_optical_flow(oldI,newI,window_size=window_size,debug_level=debug_level-1, mask=mask)
		finalU += cv2.resize(u, (oldImage.shape[1],oldImage.shape[0]), interpolation = cv2.INTER_LINEAR)
		finalV += cv2.resize(v, (oldImage.shape[1],oldImage.shape[0]), interpolation = cv2.INTER_LINEAR)
	return (finalU, finalV)
	
def load_video(subDirectory, fileName):
	full_path = os.path.join(subDirectory, fileName)
	return cv2.VideoCapture(full_path)
	
def plot_flow(oldImage, newImage, flow):
	plt.subplot(1,5,1), plt.xticks([]), plt.yticks([]), plt.title("old"), plt.imshow(oldImage, cmap='gray')
	plt.subplot(1,5,2), plt.xticks([]), plt.yticks([]), plt.title("new"), plt.imshow(newImage, cmap='gray')
	plt.subplot(1,5,3), plt.xticks([]), plt.yticks([]), plt.title("u"), plt.imshow(flow[0], cmap='gray')
	plt.subplot(1,5,4), plt.xticks([]), plt.yticks([]), plt.title("v"), plt.imshow(flow[1], cmap='gray')
	temp = np.sqrt(np.add(np.square(flow[0]),np.square(flow[1])))
	plt.subplot(1,5,5), plt.xticks([]), plt.yticks([]), plt.title("sqrt(u^2+v^2)"), plt.imshow(temp, cmap='gray')
	plt.show()
	
def plot_image(image, title=""):
	plt.xticks([]), plt.yticks([]), plt.title(title), plt.imshow(image, cmap='gray')
	plt.show()
	
def go_through_flow(video, skipframes=0, startframe=0, maxframes=10, debug_level=0, threshold=5e-3, exportFlowVideo=False):
	if (video.isOpened()):
		for i in range(startframe):
			ret,frame = video.read()
			
		#get old frame data
		ret,frame = video.read()
		old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#set desired shape of image based on which side is longer
		if old_frame.shape[0]>old_frame.shape[1]:
			shape = (1280,720)
		else:
			shape = (720,1280)
		
		#scale old_frame down to desired shape
		old_frame = cv2.resize(old_frame, shape, interpolation = cv2.INTER_LINEAR)
		
		#matricies to hold image fill data
		final_image = np.zeros(shape, dtype=int)
		set_pixels = np.zeros(shape, dtype=int)			#used to track which pixels are set
		test_pixels = np.full(shape, 1, dtype=int)		#used to compare with set_pixels
		
		frame_counter = 0
		
		if exportFlowVideo:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			xFlow = cv2.VideoWriter('xFlow.avi',fourcc, 20.0, shape)
			yFlow = cv2.VideoWriter('yFlow.avi',fourcc, 20.0, shape)
			magFlow = cv2.VideoWriter('magFlow.avi',fourcc, 20.0, shape)
		
		while(video.isOpened()):
			for i in range(skipframes):
				ret,frame = video.read()
			
			if frame_counter == maxframes or np.array_equal(set_pixels, test_pixels) or not video.isOpened():
				plot_image(final_image)
				if exportFlowVideo:
					xFlow.release()
					yFlow.release()
					magFlow.release()
				return
			
			ret, frame = video.read()
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			new_frame = cv2.resize(new_frame, shape, interpolation = cv2.INTER_LINEAR)
			
			u,v = get_optical_flow_triangle(old_frame,new_frame,debug_level=debug_level-1,mask=None)
			mag = np.sqrt(np.add(np.square(u),np.square(v)))
			
			if exportFlowVideo:
					xFlow.write(u)
					yFlow.write(v)
					magFlow.write(mag)
			
			if debug_level>0:
				plot_flow(old_frame,new_frame,(u,v))

			for i in range(shape[0]):
				for j in range(shape[1]): 
					if mag[i,j]<threshold:
						final_image[i,j] = old_frame[i,j]
						set_pixels[i,j] = 1
			
			old_frame = new_frame
			frame_counter += 1
			
people = load_video("data","people1.mp4")
go_through_flow(people, skipframes=0, startframe=33, maxframes=10, debug_level=0,exportFlowVideo=True)

