from filepicker import *  # SQ*
#from eye_tracker_helpers import *
import pandas as pd
import os
from matplotlib import pyplot as plt
import csv
import copy
from autodetection_tools import *
from tailfit9 import *
from freq_ampli_tail import *
import winsound

if __name__ == "__main__":

    ### USER INPUT HERE ###
    # folder = 'G:\DT-data\\2018\May\May 16\\1st_1st recording'  # this should be the folder where you put all the videos for analysis # folder = pickdir() # alternative way to do

    folder = 'C:\\Users\\Semmelhack Lab\\Desktop\\Binocular Lateral Version\\Circ + Rot\\20 Left\\'
    tail_fitting = 'y'  # y for doing the tailfitting
    tail_startpoint = None

    # READ ALL THE AVIS FILES WITHIN THAT FOLDER
    filenames = os.listdir(folder)
    avis = [filename for filename in filenames if os.path.splitext(filename)[1] == '.avi']

    for avi in avis:
        print '*****************************************************************************************************************************************************************************************************************************************'
        print 'current processing is: ', avi  # tell the user which avi is processing

        if tail_fitting == 'y':
            # display = askyesno(text='Display frames?')
            display = True
            displayonlyfirst = True

            'TAIL FITTTING'
            video_path = str(folder + '\\' + avi)
            if str(type(tail_startpoint)) == "<type 'NoneType'>":
                # you can either set the startpoint or process the same batch of videos with the startpoint setted in the first videos
                tail_startpoint, tail_amplitude_list = tailfit_batch([video_path], display, displayonlyfirst,
                                                                     shelve_path=folder + '\\' + avi + '.shv',
                                                                     reuse_startpoint=True)
                # tail_ampitude_list actually stores the absolute value for tail movement

            else:
                display = True
                tail_startpoint, tail_amplitude_list = tailfit_batch([video_path], display, displayonlyfirst,
                                                                     shelve_path=folder + '\\' + avi + '.shv',
                                                                     reuse_startpoint=True,
                                                                     tail_startpoint=tail_startpoint)
            # the corresponding shv files will be saved to the same folder as the tailfit
            # tail_ampitude_list actually stores the absolute value for tail movement

print 'FINISHED ANALYZING TAIL'
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 5000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)