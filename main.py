import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
	
def get_optical_flow(Old_Image, New_Image, window_size=9, debug_level=0, mask=None):
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
						x = np.linalg.pinv(A.T @ A) @ A.T @ b
						u[i,j] = x[0]
						v[i,j] = x[1]
	return (u,v)

def get_optical_flow_triangle(oldImage, newImage, window_size=9, depth=6, debug_level=0, mask=None):
	oldI = oldImage
	newI = newImage
	n_w = window_size
	finalU,finalV = get_optical_flow(oldI,newI,window_size=n_w,debug_level=debug_level-1,mask=mask)
	for i in range(depth):
		oldI = cv2.pyrDown(oldI)
		newI = cv2.pyrDown(newI)
		
		if mask is not None:
			mask = np.floor(cv2.resize(np.array(mask,dtype=float), (int(oldI.shape[1]),int(oldI.shape[0])), interpolation = cv2.INTER_LINEAR))
		
		n_w = int(n_w*(2/3))
		if n_w%2 == 0:
			n_w += 1
		
		u, v = get_optical_flow(oldI,newI,window_size=n_w,debug_level=debug_level-1, mask=mask)
		
		while oldImage.shape[0]/u.shape[0]>1.0:
			u = cv2.pyrUp(u)
		while oldImage.shape[0]/v.shape[0]>1.0:
			v = cv2.pyrUp(v)
		
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
			shape = (int(1280/2),int(720/2))
		else:
			shape = (int(720/2),int(1280/2))
		
		#scale old_frame down to desired shape
		old_frame = cv2.resize(old_frame, (shape[1],shape[0]), interpolation = cv2.INTER_LINEAR)
		
		#matricies to hold image fill data
		final_image = np.zeros(shape, dtype=int)
		set_pixels = np.full(shape, 1.0, dtype=float)
		
		previos_mags = np.zeros((5,shape[0],shape[1]))
		previos_frames = np.zeros((3,shape[0],shape[1]))
		
		frame_counter = 0
		
		max_val = -1.0
		
		while(video.isOpened()):
			for i in range(skipframes):
				ret,frame = video.read()
				if frame is None:
					break
			
			if frame is not None:
				ret, frame = video.read()
			
			if ( maxframes!=-1 and frame_counter == maxframes) or not video.isOpened() or frame is None:
				plot_image(final_image)
				plot_image(set_pixels)
				return
				
			new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			new_frame = cv2.resize(new_frame, (shape[1],shape[0]), interpolation = cv2.INTER_LINEAR)
			
			print("On flow "+str(frame_counter))
			
			u,v = get_optical_flow_triangle(old_frame,new_frame,debug_level=debug_level-1,mask=None,window_size=window_size)
			mag = np.sqrt(np.add(np.square(u),np.square(v)))
			
			if exportFlowVideo:
					
					#u_cv = np.uint8(((u+u.min())*255)/u.max())
					#v_cv = np.uint8(((v+v.min())*255)/v.max())
					#mag_cv = np.resize(np.uint8((mag*255)/mag.max()),(shape[0],shape[1],3))
					
					#xFlow.write(u_cv)
					#yFlow.write(v_cv)
					#magFlow.write(mag_cv)
					if max_val == -1.0:
						max_val = mag.sum()/(mag.shape[0]*mag.shape[1])
					
					im = Image.fromarray(np.uint8((mag*255)/mag.max())).convert('RGB')
					sim = Image.fromarray(np.uint8((np.clip(mag,0,max_val*2)*255)/max_val*2)).convert('RGB')
					im.save(os.path.join("data","video","mag"+str(frame_counter)+".jpeg"))
					sim.save(os.path.join("data","video","smag"+str(frame_counter)+".jpeg"))
			
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
			
			old_frame = new_frame
			frame_counter += 1
			
video = load_video("data","people_3.mp4")
go_through_flow(video, skipframes=0, startframe=502, maxframes=-1, window_size=9, debug_level=0, exportFlowVideo=True)

