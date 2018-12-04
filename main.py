import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
	
def get_optical_flow(Old_Image, New_Image, window_size=9, debug_level=0):
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	
	Old_Image = Old_Image / 255. # normalize pixels
	New_Image = New_Image / 255. # normalize pixels
	w = int(window_size/2)
	u=np.zeros(Old_Image.shape)
	v=np.zeros(Old_Image.shape)
	
	Ix = cv2.filter2D(Old_Image,-1,kernel_x)
	Iy = cv2.filter2D(Old_Image,-1,kernel_y)
	It = New_Image - Old_Image
	
	if debug_level>0:
		plot_image(Ix, title="partial x derivative of old image")
		plot_image(Iy, title="partial y derivative of old image")
		plot_image(It, title="partial time derivative of images")
		
	paramaters = np.zeros(Old_Image.shape+(5,))
	paramaters[..., 0] = Ix ** 2
	paramaters[..., 1] = Iy ** 2
	paramaters[..., 2] = Ix * Iy
	paramaters[..., 3] = Ix * It
	paramaters[..., 4] = Iy * It
	del Ix, Iy, It
	
	paramaters_cumulative_sums = np.cumsum(np.cumsum(paramaters, axis=0),axis=1)
	del paramaters
	
	window_sums = (paramaters_cumulative_sums[2*w+1:, 2*w+1:] - \
		paramaters_cumulative_sums[2*w+1:, :-1-2*w] - \
		paramaters_cumulative_sums[:-1-2*w, 2*w+1:] + \
		paramaters_cumulative_sums[:-1-2*w, :-1-2*w])
	del paramaters_cumulative_sums
	
	det = (window_sums[...,0]*window_sums[..., 1]-window_sums[..., 2]**2)
	
	flow_u = np.where(det != 0, \
		(-window_sums[..., 1] * window_sums[..., 3] + \
		window_sums[..., 2] * window_sums[..., 4]) / det, \
		1.0)
	flow_v = np.where(det != 0, \
		(-window_sums[..., 0] * window_sums[..., 4] + \
		window_sums[..., 2] * window_sums[..., 3]) / det, \
		1.0)
	del det
	
	u = np.pad(flow_u, [(w, w+1), (w+1, w)], mode='constant', constant_values=0)
	v = np.pad(flow_v, [(w, w+1), (w+1, w)], mode='constant', constant_values=0)
	del flow_u,flow_v

	return (u,v)

def get_optical_flow_triangle(oldImage, newImage, window_size=9, depth=4, debug_level=0):
	oldI = oldImage
	newI = newImage
	finalU,finalV = get_optical_flow(oldI,newI,window_size=window_size,debug_level=debug_level-1)
	for i in range(depth):
		oldI = cv2.pyrDown(oldI)
		newI = cv2.pyrDown(newI)
		
		window_size=int(window_size/2)
		if (window_size%2 == 0):
			window_size+=1
		if (window_size<3):
			window_size=3
		
		u, v = get_optical_flow(oldI,newI,window_size=window_size,debug_level=debug_level-1)
		
		while oldImage.shape[0]/u.shape[0]>1.0:
			u = cv2.pyrUp(u)
		while oldImage.shape[0]/v.shape[0]>1.0:
			v = cv2.pyrUp(v)
			
		finalU += 1/2*cv2.resize(u, (oldImage.shape[1],oldImage.shape[0]), interpolation = cv2.INTER_LINEAR)
		finalV += 1/2*cv2.resize(v, (oldImage.shape[1],oldImage.shape[0]), interpolation = cv2.INTER_LINEAR)
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
	
def go_through_flow(video, skipframes=0, startframe=0, maxframes=10, debug_level=0, threshold=5e-3, exportFlowVideo=False, window_size=9):
	if (video.isOpened()):
		for i in range(startframe):
			ret,frame = video.read()
			if frame is None:
				return
			
		#get old frame data
		ret,frame = video.read()
		if frame is None:
			return
		old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#set desired shape of image based on which side is longer
		if old_frame.shape[0]>old_frame.shape[1]:
			shape = (int(1280*2),int(720*2))
		else:
			shape = (int(720*2),int(1280*2))
		
		#scale old_frame down to desired shape
		old_frame = cv2.resize(old_frame, (shape[1],shape[0]), interpolation = cv2.INTER_LINEAR)
		
		#matricies to hold image fill data
		final_image = np.zeros(shape, dtype=int)
		set_pixels = np.full(shape, 1.0, dtype=float)
		
		previos_mags = np.zeros((5,shape[0],shape[1]))
		previos_frames = np.zeros((3,shape[0],shape[1]))
		
		frame_counter = 0
		max_set_val = -1.0
		
		while(video.isOpened()):
			for i in range(skipframes):
				ret,frame = video.read()
				if frame is None:
					break
			
			if frame is not None:
				ret, frame = video.read()
			
			if ( maxframes!=-1 and frame_counter == maxframes) or not video.isOpened() or frame is None:
				plot_image(final_image, title="Produced Image")
				plot_image(set_pixels, title="Cofidence")
				return
				
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			new_frame = cv2.resize(new_frame, (shape[1],shape[0]), interpolation = cv2.INTER_LINEAR)
			
			print("On flow "+str(frame_counter))
			
			feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
			
			u,v = get_optical_flow_triangle(old_frame,new_frame,debug_level=debug_level-1,window_size=window_size)
			mag = np.sqrt(np.add(np.square(u),np.square(v)))
			
			
			
			if debug_level>0:
				plot_flow(old_frame,new_frame,(u,v))
				
			previos_mags[frame_counter%5] = mag
			previos_frames[frame_counter%3] = old_frame
			
			if frame_counter>=4:
			
				magn = np.zeros(shape)
				magn += (1)*previos_mags[(frame_counter)%5]
				magn += (2)*previos_mags[(frame_counter-1)%5]
				magn += (3)*previos_mags[(frame_counter-2)%5]
				magn += (2)*previos_mags[(frame_counter-3)%5]
				magn += (1)*previos_mags[(frame_counter-4)%5]
				magn /= 12
				
				for i in range(shape[0]):
					for j in range(shape[1]): 
						if magn[i,j]<set_pixels[i,j]:
							final_image[i,j] = previos_frames[(frame_counter+1)%3][i,j]
							set_pixels[i,j] = magn[i,j]
			
				if exportFlowVideo:
					if max_set_val == -1.0:
						max_set_val = set_pixels.max()
						
					flowoo = np.zeros(shape + (3,), dtype=np.uint8)
					flowii = np.uint8(np.array([np.clip(u*125,0,255),np.abs(np.clip(u*125,-255,0)),mag*0]))
					flowoo[:,:,0] = flowii[0,:,:]
					flowoo[:,:,1] = flowii[1,:,:]
					flowoo[:,:,2] = flowii[2,:,:]
					magl = Image.fromarray(flowoo,'RGB')
					magl.save(os.path.join("data","video","u"+str(frame_counter)+".jpeg"))
					
					flowii = np.uint8(np.array([np.clip(v*125,0,255),np.abs(np.clip(v*125,-255,0)),mag*0]))
					flowoo[:,:,0] = flowii[0,:,:]
					flowoo[:,:,1] = flowii[1,:,:]
					flowoo[:,:,2] = flowii[2,:,:]
					magl = Image.fromarray(flowoo,'RGB')
					magl.save(os.path.join("data","video","v"+str(frame_counter)+".jpeg"))
					
					flowii = np.uint8(np.array([np.clip(mag*125,0,255),np.clip(mag*125,0,255),np.clip(mag*125,0,255)]))
					flowoo[:,:,0] = flowii[0,:,:]
					flowoo[:,:,1] = flowii[1,:,:]
					flowoo[:,:,2] = flowii[2,:,:]
					magl = Image.fromarray(flowoo,'RGB')
					magl.save(os.path.join("data","video","mag"+str(frame_counter)+".jpeg"))
					
					
					con = Image.fromarray(np.uint8((set_pixels*255)/max_set_val)).convert('RGB')
					fin = Image.fromarray(np.uint8(final_image)).convert('RGB')
					con.save(os.path.join("data","video","con"+str(frame_counter)+".jpeg"))
					fin.save(os.path.join("data","video","fin"+str(frame_counter)+".jpeg"))
			
			old_frame = new_frame
			frame_counter += 1
			
video = load_video("data","morepeople.mp4")
go_through_flow(video, skipframes=0, startframe=10, maxframes=100, window_size=7, debug_level=0, exportFlowVideo=True)

