import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
	
def lk_flow_fast(Old_Image, New_Image, window_size=9, debug_level=0):
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
	
	det = np.nan_to_num(det)
	#plot_image(det)
	#np.seterr(all='warn', invalid='print')
	
	flow_u = np.where(det != 0, \
		(-window_sums[..., 1] * window_sums[..., 3] + \
		window_sums[..., 2] * window_sums[..., 4]) / det, \
		0.0)
	flow_v = np.where(det != 0, \
		(-window_sums[..., 0] * window_sums[..., 4] + \
		window_sums[..., 2] * window_sums[..., 3]) / det, \
		0.0)
	del det
	
	u = np.pad(flow_u, [(w, w+1), (w+1, w)], mode='constant', constant_values=0)
	v = np.pad(flow_v, [(w, w+1), (w+1, w)], mode='constant', constant_values=0)
	del flow_u,flow_v

	return (u,v)
	
def lk_flow_improved(Old_Image, New_Image, window_size, debug_level=0):
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	
	w = int(window_size/2)
	u=np.zeros(Old_Image.shape)
	v=np.zeros(Old_Image.shape)
	
	Ix = np.array(cv2.filter2D(np.array(Old_Image, dtype='float'),-1,kernel_x), dtype='float')
	Iy = np.array(cv2.filter2D(np.array(Old_Image, dtype='float'),-1,kernel_y), dtype='float')
	It = np.array(New_Image, dtype='float') - np.array(Old_Image, dtype='float')
	
	if debug_level>0:
		plot_image(Ix, title="partial x derivative of old image")
		plot_image(Iy, title="partial y derivative of old image")
		plot_image(It, title="partial time derivative of images")
		
	paramaters = np.zeros(Old_Image.shape+(5,))
	paramaters[..., 0] = cv2.GaussianBlur(np.array(Ix ** 2, dtype='float'),(window_size,window_size),0)
	paramaters[..., 1] = cv2.GaussianBlur(np.array(Iy ** 2, dtype='float'),(window_size,window_size),0)
	paramaters[..., 2] = cv2.GaussianBlur(np.array(Ix * Iy, dtype='float'),(window_size,window_size),0)
	paramaters[..., 3] = cv2.GaussianBlur(np.array(Ix * It, dtype='float'),(window_size,window_size),0)
	paramaters[..., 4] = cv2.GaussianBlur(np.array(Iy * It, dtype='float'),(window_size,window_size),0)
	del Ix, Iy, It

	
	temp = np.nan_to_num(np.array(paramaters[..., 0]*paramaters[..., 1]-paramaters[..., 2]*paramaters[..., 2]))
	ATA = np.nan_to_num(np.array([[paramaters[..., 0],paramaters[..., 2]],[paramaters[..., 2],paramaters[..., 1]]]))
	eigns = np.nan_to_num(np.linalg.eigvals(ATA.T).T)
	
	if False:
		test = np.where((eigns[0:,:] < eigns[1,:,:]), 1.0, 0.0)
		test2 = np.where((eigns[0:,:] > eigns[1,:,:]), 1.0, 0.0)
		plot_image(test[0])
		plot_image(test2[0])
	
	
	mineign = np.minimum(eigns[0,:,:],eigns[1,:,:]) 
	maxeign = np.maximum(eigns[0,:,:],eigns[1,:,:]) 
	
	
	u = np.where( ( (mineign > 1e-2) & ( (maxeign / mineign) < 255) ), \
		paramaters[..., 4] * paramaters[..., 2] - paramaters[..., 3] * paramaters[..., 1], \
		0.0)
	v = np.where( ( (mineign > 1e-2) & ( (maxeign / mineign) < 255) ), \
		-paramaters[..., 3] * paramaters[..., 2] - paramaters[..., 4] * paramaters[..., 0], \
		0.0)
	
	u = np.nan_to_num(u/temp)
	v = np.nan_to_num(v/temp)
	
	return (u,v)

def generate_image_pyramid(image, minSize=(30,30)):
	copyImage = image
	pyramids = []
	while copyImage.shape[0] > minSize[1] and copyImage.shape[1] > minSize[0]:
		pyramids.insert(0, copyImage)
		copyImage = cv2.pyrDown(copyImage)
	return pyramids
	
def generate_window_size_pyramid(windowSize, depth, minSize=3):
	sizes = []
	for _ in range(depth):
		if int(windowSize)%2 == 0:
			windowSize = int(windowSize)+1
		sizes.insert(0, windowSize)
		windowSize = int(windowSize/2)
		if windowSize<minSize:
			windowSize=minSize
	return sizes
	
def flow_warp(image, u, v):
	u = np.array(cv2.pyrUp(u, dstsize=(image.shape[1],image.shape[0])), dtype=np.float32)
	v = np.array(cv2.pyrUp(v, dstsize=(image.shape[1],image.shape[0])), dtype=np.float32)
	return cv2.remap(image,u,v,cv2.INTER_LINEAR)
	
def coarse_to_fine_lk_flow(oldImage, newImage, window_size=9, debug_level=0):
	oldImagePyramids = generate_image_pyramid(oldImage)
	newImagePyramids = generate_image_pyramid(newImage)
	windowSizePyramid = generate_window_size_pyramid(window_size, len(oldImagePyramids)) 
	finalU = np.zeros(oldImage.shape)
	finalV = np.zeros(oldImage.shape)
	for i in range(len(oldImagePyramids)):
		
		u,v = lk_flow_improved(oldImagePyramids[i],newImagePyramids[i],windowSizePyramid[i],debug_level=debug_level-1)
		plot_image(oldImagePyramids[i])
		plot_image(newImagePyramids[i])
		plot_image(u)
		plot_image(v)
		
		if i+1 != len(oldImagePyramids):
			oldImagePyramids[i+1]=flow_warp(oldImagePyramids[i+1],u,v)
			
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
	plt.xticks([]), plt.yticks([]), plt.title(title), plt.imshow(image)
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
			shape = (int(1280),int(720))
		else:
			shape = (int(720),int(1280))
		#shape = old_frame.shape
		
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
				plot_image(final_image, title="Produced Image", cmap='gray')
				plot_image(set_pixels, title="Cofidence", cmap='gray')
				return
				
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			new_frame = cv2.resize(new_frame, (shape[1],shape[0]), interpolation = cv2.INTER_LINEAR)
			
			print("On flow "+str(frame_counter))
			
			feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
			
			u,v = coarse_to_fine_lk_flow(old_frame,new_frame,debug_level=debug_level-1,window_size=window_size)
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
						
					flowenc = np.zeros(shape + (3,), dtype=np.uint8)
					flowenc[:,:,0] = np.arctan(v,u)*57.2958
					flowenc[:,:,1] = np.clip(mag*125,0,255)
					flowenc[:,:,2] = 200
					encodedFlow = Image.fromarray(flowenc,'HSV').convert('RGB')
					encodedFlow.save(os.path.join("data","video","flow"+str(frame_counter)+".jpeg"))
					
					#con = Image.fromarray(np.uint8((set_pixels*255)/max_set_val)).convert('RGB')
					final = Image.fromarray(np.uint8(final_image)).convert('RGB')
					#con.save(os.path.join("data","video","confidence"+str(frame_counter)+".jpeg"))
					final.save(os.path.join("data","video","final"+str(frame_counter)+".jpeg"))
			else:
				if exportFlowVideo:
					flowenc = np.zeros(shape + (3,), dtype=np.uint8)
					flowenc[:,:,0] = np.arctan(v,u)*57.2958
					flowenc[:,:,1] = np.clip(mag*125,0,255)
					flowenc[:,:,2] = 200
					encodedFlow = Image.fromarray(flowenc,'HSV').convert('RGB')
					encodedFlow.save(os.path.join("data","video","flow"+str(frame_counter)+".jpeg"))
			
			old_frame = new_frame
			frame_counter += 1
			
video = load_video("data","morepeople.mp4")
video2 = load_video("data","simple2.avi")
go_through_flow(video, skipframes=0, startframe=0, maxframes=200, window_size=21, debug_level=0, exportFlowVideo=True)

