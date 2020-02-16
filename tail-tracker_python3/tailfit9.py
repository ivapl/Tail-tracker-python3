import copy
import cv2
import time
import numpy as np
import pylab, os
import pdb
from PIL import Image
from filepicker import *
import scipy.ndimage
import scipy.stats
from tailfitresult import *
from inspect import getmembers, isclass, isfunction, getmodulename, getmodule
from bouts import *
from scipy import fft
import csv
import pandas as pd
from matplotlib import pyplot as plt

dire = 'down'
__version__ = '0.9.3'

def normalizetailfit(tailfit):
    """Takes in a tailfit, and returns a normalized version which is goes from 0 to 1 normalized to taillength averaged over the first few frames
    """
    # Note. Just remind you what tailfit is:
    # NOTE. CONFIRMED. fitted_tail is the final result of the whole program.
    # NOTE. CONFIRMED. it is a list, storing arrays with the total number of total frame analyzed/read
    # NOTE. CONFIRMED. each correspond to one frame, storing x number of points (x=tail_lengh/count)
    # NOTE. CONFIRMED. points is the fitted result_point(mid point of tail edge), it is the coordinate in each frame
    # NOTE. CONFIRMED. count would be the final number of total circles.

    tail_length = (tailfit[0][-1,:]-tailfit[0][0,:]).max()   #tag
    # Question. difficult why calculate tail_length in such way?

    return [(frame-frame[0,:]).astype(float)/float(tail_length) for frame in tailfit]
    # Question. why normal this way?
    # Question. whether this return could normalize to 0 to 1? doubt it.
    # ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True).  Copy of the array, cast to a specified type.

def freql(tailfit):
    """Low frequency power of tail angles"""
    angles = tail2angles(tailfit)
    n = len(angles) # length of the signal
    Y = fft(angles)/n # fft computing and normalization

    Freq = fft(angles)
    FreqM = []
    for i in Freq:
        FreqM.append(abs(i))

        meanfreq = np.mean(Freq)
    return meanfreq

def freqm(tailfit):
    """Medium frequency power of tail angles"""
    angles = tail2angles(tailfit)
    n = len(angles) # length of the signal
    Y = fft(angles)/n # fft computing and normalization
    Y = Y[range(n/2)]

    return Y[3:6].mean()

def maxangle(tailfit):
    """Maximum tail angle"""
    return np.absolute(tail2angles(tailfit)).max()

def hashfile(afile, hasher, blocksize=65536):
    # shahash = hashfile(open(videopath, 'rb'), hashlib.sha256())
    buf = afile.read(blocksize)
    # Question. in which library is read defined?
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    return hasher.digest()

def normalizetailfit(tailfit):
    """Takes in a tailfit, and returns a normalized version which is goes from 0 to 1 normalized to taillength averaged over the first few frames
    """
    tail_length = (tailfit[0][-1,:]-tailfit[0][0,:]).max()
    return [(frame-frame[0,:]).astype(float)/tail_length for frame in tailfit]
    #could angular adjust, and maybe take average of some non-moving frames

def sliding_average(somelist, window_size = 10):
    somelistpadded = np.lib.pad(somelist,(window_size/2,window_size/2),'edge')
    return np.convolve(somelistpadded, np.ones(int(window_size))/float(window_size),mode='valid')

def sliding_gauss(somelist, window_size = 10,sigma=3):
    somelistpadded = np.lib.pad(somelist,(window_size/2,window_size/2),'edge')
    normpdf = scipy.stats.norm.pdf(range(-int(window_size/2),int(window_size/2)),0,sigma)
    return np.convolve(somelistpadded,  normpdf/np.sum(normpdf),mode='valid')[:len(somelist)]

def handleclick(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        param[0]=x
        param[1]=y


def tail_func2(x, mu, sigma, scale, offset):
    # Question. what does this function means...
    # A. seems to be a Gaussian?
    return scale * np.exp(-(x-mu)**4/(2.0*sigma**2))**.2 + offset #

##################################################################
def zeroone(thing):
    return (thing-thing.min())/(np.percentile(thing,99)-thing.min())
def scalesize(frame, multiple):
    return cv2.resize(frame,(frame.shape[0]*multiple,frame.shape[1]*multiple))
#######

def tailfit(filename,display=None,start_point=None, direction=dire, output_jpegs = False, plotlengths = False, tail_startpoint = None):

    '''
    Takes an avi filepath, fits the tail of the fish
    Display sets if the fit is shown as it is processed (slower)
    Start point is where fitting begins, if None the user is queried
    Direction is which direction the fit happens
    '''

    '''1ST PART. INITIATE THE PARAMETERS AND READ THE FRAME'''
    directions={"up":[0,-1],"down":[0,1],"left":[-1,0],"right":[1,0]}
    fitted_tail=[]

    cap = cv2.VideoCapture(filename)  ########DT error here...tag
    if not cap.isOpened():
        print("Error with video or path!")
        raise Exception('Issues opening video file!')

    frame=cap.read()[1]
    frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)  #resize-tag #resize the frame!
    cv2.destroyAllWindows()

    max_points = 200 # mostly in case it somehow gets stuck in a loop, and to preallocate the result array

    frame_fit=np.zeros((max_points,2))
    first_frame=True
    widths, convolveresults = [],[]
    test,slices = [], []

    '''2ND PART. ANALYSIS FRAME ONE BY ONE'''
    while type(frame) != type(None):

        if display:
        # display in main-function is boolean
            frame_display=frame.copy()
        if direction:
            guess_vector = np.array(directions[direction])
            # guess_vector represent which direction the fit happens
        else:
            raise Exception('Need to define a direction!') #could ask here

        '''2-1. IF FIRST FRAME'''
        '''This 2-1. session is only implemented one time during the 1st frame'''
        if first_frame:

            '''2-1.1 SET THE STARTPOINT'''
            #SET THE STARTPOINT. if we don't have a start point, query user for one
            if type(start_point)==type(np.array([])) or type(start_point) is list:
                current = np.array(start_point)
                point = current
            elif type(tail_startpoint) == type(np.array([])):
                start_point = tail_startpoint
                point = start_point
                current = start_point
            else:
                handlec=handleclick
                cv2.namedWindow('first')
                cv2.imshow("first",frame)
                cv2.moveWindow('first',0,0)

                cv2.waitKey(10)
                point = np.array([-1,-1])
                cv2.setMouseCallback("first",handlec,point)
                # cv2.setMouseCallback(windowName, onMouse[, param])
                print("Click on start of the fish's tail")
                cv2.waitKey(10)  # Question. difference between 0 and 10?    #tag
                while (point == np.array([-1,-1])).all(): 
                    cv2.waitKey(10)
                current = point
                start_point = current
                print('start point is ', start_point)
                cv2.destroyWindow('first')
            '''2-1.2 ILLUMINATION ANALYSIS FOR BG & FISH'''
            # BUILD THE HISTOGRAM, frame is np.ndarray, 2D-gray scale, 3D-RGB
            if frame.ndim == 2:
                hist = np.histogram(frame[:,:],10,(0,255))
            elif frame.ndim == 3:
                hist = np.histogram(frame[:,:,0],10,(0,255))
            else:
                raise Exception('Unknown video format!')

            background = hist[1][hist[0].argmax()]/2+hist[1][min(hist[0].argmax()+1,len(hist[0]))]/2
            if frame.ndim == 2:
                fish = frame[point[1]-2:point[1]+2,point[0]-2:point[0]+2].mean()
            elif frame.ndim == 3:
                fish = frame[point[1]-2:point[1]+2,point[0]-2:point[0]+2,0].mean()


            '''2-1.3 BUILD THE GAUSSIAN KERNEL & SET DISPLAY '''
            print("Starting tailfit on:  ", filename)
            FPS = cap.get(cv2.CAP_PROP_FPS)
            numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            guess_line_width = 30

            normpdf = scipy.stats.norm.pdf(np.arange((-guess_line_width+1)/4+1,(guess_line_width-1)/4),0,8)

            if display:
                cv2.namedWindow("frame_display")
                cv2.moveWindow("frame_display",0,0)

            starttime = time.time()

        else:         #task. temporarily skip
            current= fitted_tail[-1][0,:]

        '''2-2. SET SPACING'''
        tailpoint_spacing = 5

        '''2-3.FIT THE TAIL WITH CIRCILES IN THIS FRAME(BIG FOR COUNT LOOPS)'''

        for count in range(max_points):
            '''2-3.1 SET THE GUESS POINT/GUESS_LINE'''

            '''2-3.1.1 GUESS IS THE NEXT FITTED POINTS'''
            if count == 0:
                guess = current
            elif count == 1:
                guess = current + guess_vector*tailpoint_spacing 
            else:
                guess_vector = guess_vector/(((guess_vector**2).sum())**.5) #normalize guess vector
                guess = current + guess_vector*tailpoint_spacing

            '''2-3.1.2 DRAW THE START AND END'''
            guess_line_start = guess + np.array([-guess_vector[1],guess_vector[0]])*guess_line_width/2
            guess_line_end = guess + np.array([guess_vector[1],-guess_vector[0]])*guess_line_width/2

            x_indices = np.int_(np.linspace(guess_line_start[0],guess_line_end[0],guess_line_width))
            y_indices = np.int_(np.linspace(guess_line_start[1],guess_line_end[1],guess_line_width))


            '''2-3.1.3 JUDGE IF THE CLIP IS PROPER'''
            if max(y_indices) >= frame.shape[0] or min(y_indices) < 0 or max(x_indices) >= frame.shape[1] or min(x_indices) < 0:

                y_indices = np.clip(y_indices,0, frame.shape[0]-1)
                x_indices = np.clip(x_indices,0, frame.shape[1]-1)
                print("Tail got too close to the edge of the frame, clipping search area!")

            '''2-3.1.4 DRAW THE GUESS_SLICE'''
            guess_slice= frame[y_indices,x_indices]


            if guess_slice.ndim == 2:
                guess_slice=guess_slice[:,0]
            else:
                guess_slice=guess_slice[:]

            '''2-3.2 BASELINE SUBSTRACTION'''
            if fish < background:
                guess_slice = (background-guess_slice)
            else:
                guess_slice = (guess_slice-background)

            slices += [guess_slice]  ######tag

            hist = np.histogram(guess_slice, 10)

            guess_slice = guess_slice-guess_slice[((hist[1][hist[0].argmax()] <= guess_slice)&(guess_slice<hist[1][hist[0].argmax()+1]))].mean()  #tag

            '''2-3.3 FILTER! SMOOTH THE GUESS_SLICE '''
            sguess = scipy.ndimage.filters.percentile_filter(guess_slice,50,5)


            '''2-3.4, 1ST FRAME-1, DELINEATE ALL THE NEWPOINT'''
            if first_frame:
                '''2-3.4.1 DEFINE THE EDGE OF TAIL AND FIND THE MID-POINT'''
                tailedges = np.where(np.diff(sguess>(sguess.max()*.25)))[0]

                if len(tailedges)>=2:
                    tailedges = tailedges-len(sguess)/2.0
                    tailindexes = tailedges[np.argsort(np.abs(tailedges))[0:2]]
                    result_index_new = (tailindexes).mean()+len(sguess)/2.0
                    widths +=[abs(tailindexes[0]-tailindexes[1])]
                else:
                    result_index_new = None
                    tail_length = count
                    break

                '''2-3.4.2 CONVOLUTION & NEWPOINT'''
                results = np.convolve(normpdf,guess_slice,"valid")
                convolveresults+=[results]
                result_index = results.argmax() - int(results.size/2+guess_slice.size/2)
                newpoint = np.array([x_indices[int(result_index_new)],y_indices[int(result_index_new)]])  #DT-
            else:       
                results= np.convolve(tailfuncs[count],guess_slice,"valid")
                result_index = results.argmax() - int(results.size/2+guess_slice.size/2)
                newpoint = np.array([x_indices[result_index],y_indices[result_index]])

            '''2-3.5, 1ST FRAME-2, FUNCTION UNKNOWN'''
            if first_frame:
                '''2-3.5.1 CHECK FITTING SESSION, BREAK IF NECCESSARY'''
                if count > 10:
                    trapz = [pylab.trapz(result-result.mean()) for result in convolveresults]   #tag
                    slicesnp = np.vstack(slices)
                    
                    if np.array(trapz[-3:]).mean() < .2:  #tag
                        tail_length = count
                        break
                    elif slicesnp[-1,result_index-2:result_index+2].mean()<5:
                        tail_length = count
                        break
                        
            elif count > tail_length*.8 and np.power(newpoint-current,2).sum()**.5 > tailpoint_spacing*1.5:
                break
            elif count == tail_length:
                break    

            '''2-3.6 DRAW THE CIRCLES ALONG THE TAIL, UPDATE VECTORS AND THEN CURRENT'''
            if display:
                cv2.circle(frame_display,(int(newpoint[0]),int(newpoint[1])),2,(0,0,0))    #tag
                # DT CODE: print 'newpoint: ', newpoint
                # Note. CONFIRMED: circle is drawed one by one, newpoint is simple list consists of two items
                # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]), returns img
                # frame_display is defined by this: frame_display=frame.copy()
##                frame_display[y_indices,x_indices]=0

            frame_fit[count,:] = newpoint
            # frame_fit=np.zeros((max_points,2))
            # put the newpoint into the frame_fit array, a 2D array

            if count>0:
                guess_vector = newpoint-current
            # Question. function of this if block?
            # A. guess_vector gives the direction of guess, current is old point

            current = newpoint 
        '''2-4. STRANGE SWIDTHS, FINALLY! JUMP OUT OF FOR-COUNT'''
        if first_frame:
        # first_frame just told the program if it is processing the first frame

            swidths = scipy.ndimage.filters.percentile_filter(widths,50,8)  #task...temporarily just keep it
            # Julie-Question. meaning of this? and width, pleasssssssssssssse
            # DT code

            swidths = np.lib.pad(swidths,[0,5],mode='edge')  #tag bug
            # Note. Bug. IndexError: index -1 is out of bounds for axis 0 with size 0
            # np.lib.pad, choose the last item of swidths and add
            # Question. why pads the fish?
            # numpy.pad(array, pad_width, mode, **kwargs), Pads an array

            tailfuncs = [tail_func2(np.arange((-guess_line_width+1)/4+1,(guess_line_width-1)/4),0, swidth, 1, 0) for swidth in swidths]  #tag
            # Note. guess_line_width = 51
            # Note. def tail_func2(x, mu, sigma, scale, offset)
            # Question. so swidth is going to be sigma? why is that????

        '''2-5. APPEND FITTED_TAIL'''
        fitted_tail.append(np.copy(frame_fit[:count]))

        '''2-6. DISPLAY THE FRAME!'''
        if display:
            cv2.putText(frame_display,str(count),(340,25),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(225,10,20) );
            cv2.putText(frame_display,str(len(fitted_tail)-1),(15,25),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(25,10,20) ); #-1 because the current frame has already been appended
            cv2.imshow("frame_display",frame_display)
            # cv2.waitKey(0)  #DT-CODE: Manual control to analyze the frame one by one
            if first_frame:
                delaytime = 1
                # Question. unit ms?
            else:
                minlen = min([fitted_tail[-2].shape[0],fitted_tail[-1].shape[0]])-1
                delaytime = int(min(max((np.abs((fitted_tail[-2][minlen,:]-fitted_tail[-1][minlen,:])**2).sum()**.5)**1.2*3-1,1), 500))
            cv2.waitKey(delaytime)

        '''2-7. OUTPUT JPEG'''
        #task. temp omit
        if output_jpegs:
            if first_frame:
                jpegs_dir = pickdir()
                if not os.path.exists(jpegs_dir):
                    os.makedirs(jpegs_dir)
            jpg_out = Image.fromarray(frame_display)
            jpg_out.save(os.path.normpath(jpegs_dir +'\\'+ str(len(fitted_tail)-1)+'.jpg'))

        '''2-8. FALSE 1ST FRAME AND READ NEXT FRAME'''
        first_frame = False
        # cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,float(len(fitted_tail)) );  #workaround for raw videos crash, but massively (ie 5x) slower
        s, frame = cap.read()
        if s:     # Only process valid image frames
           frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)  #resize-tag #resize the frame!

        # turn off the first_frame and update the frame with next frame in video
        # cv2.VideoCapture.read([image]) returns tuple (retval, image), so frame only takes the returned image
    print("Done in %.2f seconds" % (time.time()-starttime))

    fit_lengths = np.array([len(i) for i in fitted_tail])   ########tag########
    if np.std(fit_lengths) > 3 or plotlengths:
        print('Abnormal variances in tail length detected, check results: ', filename)
        pylab.plot(range(0,len(fitted_tail)),fit_lengths)
        pylab.ylim((0,5+max(fit_lengths)))
        pylab.xlabel("Frame")
        pylab.ylabel('Tail points')
        pylab.title('Tail fit lengths')
        print('Close graph to continue!')
        pylab.show()

    if any(fit_lengths<25):
        print("Warning - short tail detected in some frames - min: ", min(fit_lengths))

    if len(fitted_tail) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        print("Warning - number of frames processed doesn't match number of video frames - can happen with videos over 2gb!")
        print("Frames processed: " , len(fitted_tail))
        print("Actual frames according to video header: " , int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    '''4TH PART. DESTROYWINDOW AND RETURN!'''
    cv2.destroyAllWindows()
    return fitted_tail, start_point, direction, FPS, numframes
    # Question. what is the function of returned value? simply just stored in shelf?

def tailfit_batch(video_list=[], display=True, displayonlyfirst = True, output = '', startpoints=None, reuse_startpoint = False, tail_startpoint = None):

    first = True
    for i, videopath in enumerate(video_list):
        if type(startpoints) is list:
            fittedtail, startpoint,  direction, FPS, numframes  = tailfit(videopath,(first or not displayonlyfirst) and display ,startpoints[i])
        else:
            fittedtail, startpoint, direction, FPS, numframes  = tailfit(videopath,(first or not displayonlyfirst) and display ,startpoints, tail_startpoint = tail_startpoint)

            if reuse_startpoint:
                startpoints = [startpoint]*len(video_list) 
        
        fittedtail = normalizetailfit(fittedtail)
        tailangle = [-i for i in tail2angles(fittedtail)]
        frame = range(1, len(tailangle) + 1)
        data = {'Frame': frame, 'Tail Angle': tailangle}
        df = pd.DataFrame(data)
        df.to_csv(output, index=False)
        plt.plot(frame, tailangle)
        plt.savefig(output.replace(".csv", ".png"))
        plt.close()
        
        if first:
            first = False
        
    return startpoint