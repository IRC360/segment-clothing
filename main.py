#Import the OpenCV and dlib libraries
import cv2
import threading
import time
import numpy as np
import dlib
#from matplotlib import pyplot as plt


#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder
faceCascade = cv2.CascadeClassifier('OpenCV-detection-models-master\haarcascades\haarcascade_frontalface_default.xml')

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 800
OUTPUT_SIZE_HEIGHT = 600
vid=True
BLUR = 21


#We are not doing really face recognition
def doRecognizePerson(faceNames, fid):
    time.sleep(2)
    faceNames[ fid ] = "Person" #+ str(fid)

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    


def detectAndTrackMultipleFaces():
    #Open the first webcame device
    capture = cv2.VideoCapture('vid/all.mp4')

    #Create two opencv named windows
    #cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    #cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)

    #Start the window thread for the two windows we are using
    cv2.startWindowThread()

    #Create the tracker we will use
    tracker = dlib.correlation_tracker()

    #The variable we use to keep track of the fact whether we are
    #currently using the dlib tracker
    trackingFace = 0

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
    #isRecording=True
    try:
        while vid :
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()
            #if(isRecording):#read the boolean to decide whether to write frame or not
               
               #Resize the image to 320x240
            baseImage = cv2.resize( fullSizeBaseImage, ( 800, 600))

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(27)
            if pressedKey == ord('q'):
                break
            

            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()

            #STEPS:
            # * Update all trackers and remove the ones that are not 
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces. 
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to dof
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id


            #Increase the framecounter
            frameCounter += 1 

            #Update all the trackers and remove the ones for which the update
            #indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )

                #If the tracking quality is good enough, we must delete
                #this tracker
                if trackingQuality < 7:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )
         
            #Every 10 frames, we will have to determine which faces
            #are present in the frame
            if (frameCounter % 10) == 0:

                #For the face detection, we need to make use of a gray
                #colored image so we will convert the baseImage to a
                #gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                #Now use the haar cascade detector to find all faces
                #in the image
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                #Loop over all faces and check if the area for this
                #face is the largest so far
                #We need to convert it to int here because of the
                #requirement of the dlib tracker. If we omit the cast to
                #int here, you will get cast errors since the detector
                #returns numpy.int32 and the tracker requires an int
                for (_x,_y,_w,_h) in faces:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    #Variable holding information which faceid we 
                    #matched with
                    matchedFid = None

                    #Now loop over all the trackers and check if the 
                    #centerpoint of the face is within the box of a 
                    #tracker
                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the 
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region 
                        #detected as a face. If both of these conditions hold
                        #we have a match
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                             ( t_y <= y_bar   <= (t_y + t_h)) and 
                             ( x   <= t_x_bar <= (x   + w  )) and 
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid


                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:

                        print("Creating new tracker " + str(currentFaceID))

                        #Create and store the tracker 
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x-5,
                                                            y-5,
                                                            x+w+5,
                                                            y+h+5))

                        
                        y2 = y+h
                        
                        new_img = cv2.rectangle(baseImage, (int(x-(x*0.2)), int(y2)), (int((x+w)+(x*0.2)), int((y2+h)+(h*1.7))), (0,255,255), 2)
                        new_img2 = cv2.rectangle(baseImage, (int(x-(x*0.2)), int(y2+(h*2.8))), (int((x+w)+(x*0.2)), int((y2+h)+h+((y2+h)*1.2))), (255,255,0), 2) 
                        upper = new_img[int(y2):int((y2+h)+(+h*1.7)), int(x-(x*0.2)):int((x+w)+(x*0.2))]
                        lower = new_img2[int(y2+(h*2.8)):int((y2+h)+h+((y2+h)*1.2)), int(x-(x*0.2)):int((x+w)+(x*0.2))] 
                        rupper = cv2.resize( upper, ( 257, 513))
                        rlower = cv2.resize( lower, ( 257, 513))
                        #cv2.imshow('test',upper)
                        #cv2.imshow("try",rlower)

                        #GrABCUT CODE
                        #upper
                        umask = np.zeros(rupper.shape[:2],np.uint8)
                        ubgdModel = np.zeros((1,65),np.float64)
                        ufgdModel = np.zeros((1,65),np.float64)
                        urect = (10,10,236,492)
                        cv2.grabCut(rupper,umask,urect,ubgdModel,ufgdModel,5,cv2.GC_INIT_WITH_RECT)
                        umask2 = np.where((umask==2)|(umask==0),0,1).astype('uint8')
                        rupper =rupper*umask2[:,:,np.newaxis]
                        cv2.imshow('Upper Grabcut',rupper)
                        #plt.imshow(rlower)
                        #plt.colorbar()
                        #plt.show()
                        lmask = np.zeros(rlower.shape[:2],np.uint8)
                        lbgdModel = np.zeros((1,65),np.float64)
                        lfgdModel = np.zeros((1,65),np.float64)
                        lrect = (10,10,236,492)
                        cv2.grabCut(rlower,lmask,lrect,ubgdModel,lfgdModel,5,cv2.GC_INIT_WITH_RECT)
                        lmask2 = np.where((lmask==2)|(lmask==0),0,1).astype('uint8')
                        rlower =rlower*lmask2[:,:,np.newaxis]
                        cv2.imshow('Lower Grabcut',rlower)
                        
                        #SKIN DETECTION CODE

                        #upper
                        uhsv = cv2.cvtColor(rupper, cv2.COLOR_BGR2HSV)
                        uskin1 = np.array([0,32,60])
                        uskin2 = np.array([42,235,255])
                        umask3 = cv2.inRange(uhsv, uskin1, uskin2)
                        urev = cv2.bitwise_not(umask3)
                        ures = cv2.bitwise_and(rupper,rupper, mask= urev)                    
                        #cv2.imshow('mask',mask3)
                        cv2.imshow('Upper Skin Removal',ures)
                        #cv2.imwrite('images/1.jpg', ures)
                        
                        lhsv = cv2.cvtColor(rlower, cv2.COLOR_BGR2HSV)
                        lskin1 = np.array([0,32,60])
                        lskin2 = np.array([42,235,255])
                        lmask3 = cv2.inRange(lhsv, lskin1, lskin2)
                        lrev = cv2.bitwise_not(lmask3)
                        lres = cv2.bitwise_and(rlower,rlower, mask= lrev)                    
                        #cv2.imshow('mask',mask3)
                        cv2.imshow('Lower Skin Removal',lres)
                        #cv2.imwrite('images/2.jpg', lres)

                        

                        #LBP CODE
                        #upper
                        uheight, uwidth, uchannel = ures.shape
                        uimg_gray = cv2.cvtColor(ures, cv2.COLOR_BGR2GRAY)
                        uimg_lbp = np.zeros((uheight, uwidth,3), np.uint8)
                        for i in range(0, uheight):
                            for j in range(0, uwidth):
                                uimg_lbp[i, j] = lbp_calculated_pixel(uimg_gray, i, j)
                        cv2.imshow('Upper LBP',uimg_lbp)

                        lheight, lwidth, lchannel = lres.shape
                        limg_gray = cv2.cvtColor(lres, cv2.COLOR_BGR2GRAY)
                        limg_lbp = np.zeros((lheight, lwidth,3), np.uint8)
                        for i in range(0, lheight):
                            for j in range(0, lwidth):
                                limg_lbp[i, j] = lbp_calculated_pixel(limg_gray, i, j)
                        cv2.imshow('Upper LBP',limg_lbp)        
                        
                        
                        
                        #SURF CODE
                        #upper
                        usurf = cv2.xfeatures2d.SURF_create()
                        ukeypoints, udescriptors = usurf.detectAndCompute(uimg_lbp, None)
                        uimg = cv2.drawKeypoints(uimg_lbp, ukeypoints, None)
                        cv2.imshow("Upper Surf", uimg)
                        cv2.imwrite('images/1.jpg', uimg)
                        
                        lsurf = cv2.xfeatures2d.SURF_create()
                        lkeypoints, udescriptors = lsurf.detectAndCompute(limg_lbp, None)
                        limg = cv2.drawKeypoints(limg_lbp, lkeypoints, None)
                        cv2.imshow("Upper Surf", limg)
                        cv2.imwrite('images/2.jpg', limg)

                        

                        #detected(body, new_img)
                        faceTrackers[ currentFaceID ] = tracker
                        #Start a new thread that is used to simulate 
                        #face recognition. This is not yet implemented in this
                        #version :)
                        t = threading.Thread( target = doRecognizePerson ,
                                               args=(faceNames, currentFaceID))
                        t.start()

                        #Increase the currentFaceID counter
                        currentFaceID += 1
                
            #Now loop over all the trackers we have and draw the rectangle
            #around the detected faces. If we 'know' the name for this person
            #(i.e. the recognition thread is finished), we print the name
            #of the person, otherwise the message indicating we are detecting
            #the name of the person
            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()
                
                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())

                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)
                y2 = t_y+t_h
            
                cv2.rectangle(resultImage, (int(t_x-(t_x*0.2)), int(y2)), (int((t_x+t_w)+(t_x*0.2)), int((y2+t_h)+(t_h*1.5))), (0,255,255), 2)
                cv2.rectangle(resultImage, (int(t_x-(t_x*0.2)), int(y2+(t_h*2.6))), (int((t_x+t_w)+(t_x*0.2)), int((y2+t_h)+t_h+((y2+t_h)*0.9))), (255,255,0), 2)

                if fid in faceNames.keys():
                    cv2.putText(resultImage, faceNames[fid] , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                #isRecording =False     
                #if pressedKey == ord('c'):#Continue
                #    isRecording=True   
                

            #Since we want to show something larger on the screen than the
            #original 320x240, we resize the image again
            #
            #Note that it would also be possible to keep the large version
            #of the baseimage and make the result image a copy of this large
            #base image and use the scaling factor to draw the rectangle
            #at the right coordinates.
            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))

            #Finally, we want to show the images on the screen
            #cv2.imshow("base-image", baseImage)
            cv2.imshow("result-image", largeResult)


    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass
    


    #Destroy any OpenCV windows and exit the application
    cv2.destroyAllWindows()
    exit(0)

if __name__ == '__main__':
    detectAndTrackMultipleFaces()