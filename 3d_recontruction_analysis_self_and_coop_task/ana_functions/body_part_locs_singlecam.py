#  function - compare the body track result from different camera pairs
def body_part_locs_singlecam(bodyparts_camN_camNM,singlecam_ana_type,animalnames_videotrack,bodypartnames_videotrack,date_tgt):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    # load camN h5 file
    bodyparts_camN_camNM_data = pd.read_hdf(bodyparts_camN_camNM)
    
    # get body part location from one single camera
    body_part_locs_notfix = {}
    body_part_locs = {}
  
    nbodies = np.shape(bodypartnames_videotrack)[0]

    for iname in animalnames_videotrack:
        for ibody in np.arange(0,nbodies,1):
            ibodyname = bodypartnames_videotrack[ibody]

            xxx = bodyparts_camN_camNM_data[(singlecam_ana_type, iname, ibodyname, 'x')]
            #
	    # fill in the nan data point
            # xxx_filled = xxx.interpolate(method='nearest',limit_direction='both')
            # xxx_filled = xxx_filled.interpolate(method='linear',limit_direction='both')
            xxx_filled = xxx
            #            
            # remove outlier
            q1 = np.nanquantile(xxx_filled,0.25)
            q3 = np.nanquantile(xxx_filled,0.75)
            thres1 = q1 - 1.5*abs(q3-q1)
            thres2 = q3 + 1.5*abs(q3-q1)
            ind = (xxx_filled>thres2) | (xxx_filled<thres1)
            xxx_filled[ind] = np.nan
            #        
            # fill in the nan data point after outlier removal
            # xxx_filled = xxx_filled.interpolate(method='linear',limit_direction='both')
            #        
            # smooth the data point   
            # std was chosen based on the data (mean std across session, animal, body part and x,y,z is ~2)
            # xxx_filtered = xxx_filled.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
            #
            xxx = xxx_filled


            yyy = bodyparts_camN_camNM_data[(singlecam_ana_type, iname, ibodyname, 'y')]
	    #
            # fill in the nan data point
            # yyy_filled = yyy.interpolate(method='nearest',limit_direction='both')
            # yyy_filled = yyy_filled.interpolate(method='linear',limit_direction='both')
            yyy_filled = yyy
            #            
            # remove outlier
            q1 = np.nanquantile(yyy_filled,0.25)
            q3 = np.nanquantile(yyy_filled,0.75)
            thres1 = q1 - 1.5*abs(q3-q1)
            thres2 = q3 + 1.5*abs(q3-q1)
            ind = (yyy_filled>thres2) | (yyy_filled<thres1)
            yyy_filled[ind] = np.nan
            #        
            # fill in the nan data point after outlier removal
            # yyy_filled = yyy_filled.interpolate(method='linear',limit_direction='both')
            #        
            # smooth the data point   
            # std was chosen based on the data (mean std across session, animal, body part and x,y,z is ~2)
            # yyy_filtered = yyy_filled.rolling(window=10, win_type='gaussian', center=True).mean(std=2)
	    #
            yyy = yyy_filled

            body_part_locs_notfix[(iname,ibodyname)] = np.transpose(np.vstack((np.array(xxx),np.array(yyy))))
 
    # fix the messed up dodson and scorch id (messed up animal1 and animal2)
    for ibody in np.arange(0,nbodies,1):
        ibodyname = bodypartnames_videotrack[ibody]
   
        if(np.nanmean(body_part_locs_notfix[('dodson',ibodyname)])>np.nanmean(body_part_locs_notfix[('scorch',ibodyname)])):
            body_part_locs[('dodson',ibodyname)] = body_part_locs_notfix[('dodson',ibodyname)]
            body_part_locs[('scorch',ibodyname)] = body_part_locs_notfix[('scorch',ibodyname)]
        else:
            body_part_locs[('dodson',ibodyname)] = body_part_locs_notfix[('scorch',ibodyname)]
            body_part_locs[('scorch',ibodyname)] = body_part_locs_notfix[('dodson',ibodyname)]
        

            
    return body_part_locs

       
   	    



        
