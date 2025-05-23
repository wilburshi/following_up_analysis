#  function - get singlecam variables and neurons and then do the GLM fitting

def get_singlecam_bhv_var_for_neuralGLM_fitting(animal1, animal2, animalnames_videotrack, session_start_time, starttime, endtime, totalsess_time, blockstarttime_all_idate, blockendtime_all_idate, force1_all_idate, force2_all_idate, stg_twins, time_point_pull1, time_point_pull2, time_point_juice1, time_point_juice2, time_point_pulls_succfail, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, gaze_thresold, spike_clusters_data, spike_time_data, spike_channels_data):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    


    # prepare some time stamp data
    # merge oneway gaze and mutual gaze 
    # note!! the time stamps are already aligned to the start of session (instead of the start of video recording)
    oneway_gaze1 = np.sort(np.hstack((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.hstack((oneway_gaze2,mutual_gaze2)))
    #
    # only choose the time between starttime and endtime
    oneway_gaze1 = oneway_gaze1[(oneway_gaze1<=endtime)&(oneway_gaze1>=starttime)]
    oneway_gaze2 = oneway_gaze2[(oneway_gaze2<=endtime)&(oneway_gaze2>=starttime)]
    #
    time_point_pull1 = np.array(time_point_pull1)
    time_point_pull1 = time_point_pull1[(time_point_pull1<=endtime)&(time_point_pull1>=starttime)]
    time_point_pull2 = np.array(time_point_pull2)
    time_point_pull2 = time_point_pull2[(time_point_pull2<=endtime)&(time_point_pull2>=starttime)]
    #
    time_point_juice1 = np.array(time_point_juice1)
    time_point_juice1 = time_point_juice1[(time_point_juice1<=endtime)&(time_point_juice1>=starttime)]
    time_point_juice2 = np.array(time_point_juice2)
    time_point_juice2 = time_point_juice2[(time_point_juice2<=endtime)&(time_point_juice2>=starttime)]

    # get the gaze start and stop
    #animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = oneway_gaze1
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
    animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
    animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
    #
    #animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = oneway_gaze2
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
    animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
    animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 

    # get the successful and failed pull time point
    time_point_pull1_succ = np.array(time_point_pulls_succfail['pull1_succ'])
    time_point_pull2_succ = np.array(time_point_pulls_succfail['pull2_succ'])
    time_point_pull1_fail = np.array(time_point_pulls_succfail['pull1_fail'])
    time_point_pull2_fail = np.array(time_point_pulls_succfail['pull2_fail'])


    # find the action that below to one of three strategies - pull in sync, social attention, gaze lead pull
    #
    # pull2 -> pull1 (no gaze involved)
    tpoint_pull2_to_pull1 = np.array([])
    tpoint_pull2_to_pull1_not = np.array([])
    for itpoint_pull1 in time_point_pull1:
        itv = itpoint_pull1 - time_point_pull2
        itv_alt = itpoint_pull1 - oneway_gaze1
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_pull2_to_pull1 = np.append(tpoint_pull2_to_pull1,[itpoint_pull1])
            else:
                tpoint_pull2_to_pull1_not = np.append(tpoint_pull2_to_pull1_not,[itpoint_pull1])
        except:
            tpoint_pull2_to_pull1_not = np.append(tpoint_pull2_to_pull1_not,[itpoint_pull1])
    # pull1 -> pull2 (no gaze involved)
    tpoint_pull1_to_pull2 = np.array([])
    tpoint_pull1_to_pull2_not = np.array([])
    for itpoint_pull2 in time_point_pull2:
        itv = itpoint_pull2 - time_point_pull1
        itv_alt = itpoint_pull2 - oneway_gaze2
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_pull1_to_pull2 = np.append(tpoint_pull1_to_pull2,[itpoint_pull2])
            else:
                tpoint_pull1_to_pull2_not = np.append(tpoint_pull1_to_pull2_not,[itpoint_pull2])
        except:
            tpoint_pull1_to_pull2_not = np.append(tpoint_pull1_to_pull2_not,[itpoint_pull2])
    # pull2 -> gaze1 (did not translate to own pull)
    tpoint_pull2_to_gaze1 = np.array([])
    tpoint_pull2_to_gaze1_not = np.array([])
    for itpoint_gaze1 in oneway_gaze1:
        itv = itpoint_gaze1 - time_point_pull2
        itv_alt = time_point_pull1 - itpoint_gaze1  
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_pull2_to_gaze1 = np.append(tpoint_pull2_to_gaze1,[itpoint_gaze1])
            else:
                tpoint_pull2_to_gaze1_not = np.append(tpoint_pull2_to_gaze1_not,[itpoint_gaze1])
        except:
            tpoint_pull2_to_gaze1_not = np.append(tpoint_pull2_to_gaze1_not,[itpoint_gaze1])
    # pull1 -> gaze2 (did not translate to own pull)
    tpoint_pull1_to_gaze2 = np.array([])
    tpoint_pull1_to_gaze2_not = np.array([])
    for itpoint_gaze2 in oneway_gaze2:
        itv = itpoint_gaze2 - time_point_pull1
        itv_alt = time_point_pull2 - itpoint_gaze2  
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_pull1_to_gaze2 = np.append(tpoint_pull1_to_gaze2,[itpoint_gaze2])
            else:
                tpoint_pull1_to_gaze2_not = np.append(tpoint_pull1_to_gaze2_not,[itpoint_gaze2])
        except:
            tpoint_pull1_to_gaze2_not = np.append(tpoint_pull1_to_gaze2_not,[itpoint_gaze2])       
    # gaze1 -> pull1 (no sync pull)
    tpoint_gaze1_to_pull1 = np.array([])
    tpoint_gaze1_to_pull1_not = np.array([])
    for itpoint_pull1 in time_point_pull1:
        itv = itpoint_pull1 - oneway_gaze1
        itv_alt = itpoint_pull1 - time_point_pull2
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_gaze1_to_pull1 = np.append(tpoint_gaze1_to_pull1,[itpoint_pull1])
            else:
                tpoint_gaze1_to_pull1_not = np.append(tpoint_gaze1_to_pull1_not,[itpoint_pull1])
        except:
            tpoint_gaze1_to_pull1_not = np.append(tpoint_gaze1_to_pull1_not,[itpoint_pull1])        
    # gaze2 -> pull2
    tpoint_gaze2_to_pull2 = np.array([])
    tpoint_gaze2_to_pull2_not = np.array([])
    for itpoint_pull2 in time_point_pull2:
        itv = itpoint_pull2 - oneway_gaze2
        itv_alt = itpoint_pull2 - time_point_pull1
        try:
            if (np.nanmin(itv[itv>0]) <= stg_twins): # & (np.nanmin(itv_alt[itv_alt>0]) > stg_twins):
                tpoint_gaze2_to_pull2 = np.append(tpoint_gaze2_to_pull2,[itpoint_pull2])
            else:
                tpoint_gaze2_to_pull2_not = np.append(tpoint_gaze2_to_pull2_not,[itpoint_pull2])
        except:
            tpoint_gaze2_to_pull2_not = np.append(tpoint_gaze2_to_pull2_not,[itpoint_pull2])
            
    
    #
    nanimals = np.shape(animalnames_videotrack)[0]

    gausKernelsize = 3

    fps = 30     


    con_vars_plot = ['leverpull_prob','socialgaze_prob','juice_prob',
                     'sync_pull_prob','gaze_lead_pull_prob','social_attention_prob','forcelevel']

    nconvarplots = np.shape(con_vars_plot)[0]

    data_summary_names = con_vars_plot

    data_summary = dict.fromkeys([animal1,animal2],[])

    # keep the tracking time to the start of video recording
    starttimeframe = int(np.round((starttime + session_start_time)*fps))
    endtimeframe = int(np.round((endtime + session_start_time)*fps)+1)
    #
    nallframes = endtimeframe-starttimeframe
    
    totalsess_nframes = int(np.round(totalsess_time*fps))
    

    # define and organize the continuous variables
    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]


        # get the variables
        # keep the tracking time to the start of video recording

        #
        # get the pull time series
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_pull = time_point_pull1+session_start_time
        elif ianimal == 1:
            timepoint_pull = time_point_pull2+session_start_time
        #
        try:
            timeseries_pull = np.zeros((totalsess_nframes,))
            timeseries_pull[list(map(int,list(np.round((timepoint_pull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_pull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_pull*fps))))+1,))
            timeseries_pull[list(map(int,list(np.round(timepoint_pull*fps))))]=1
        # leverpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_pull,gausKernelsize)
        leverpull_prob = timeseries_pull
        # 

        #
        # get the juice time series
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_juice = time_point_juice1+session_start_time
        elif ianimal == 1:
            timepoint_juice = time_point_juice2+session_start_time
        #
        try:
            timeseries_juice = np.zeros((totalsess_nframes,))
            timeseries_juice[list(map(int,list(np.round((timepoint_juice))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_juice = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_juice*fps))))+1,))
            timeseries_juice[list(map(int,list(np.round(timepoint_juice*fps))))]=1
        # juice_prob = scipy.ndimage.gaussian_filter1d(timeseries_juice,gausKernelsize)
        juice_prob = timeseries_juice
        # 

        #
        # get the self social gaze time series
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_gaze = oneway_gaze1+session_start_time
        elif ianimal == 1:
            timepoint_gaze = oneway_gaze2+session_start_time
        #
        try:
            timeseries_gaze = np.zeros((totalsess_nframes,))
            timeseries_gaze[list(map(int,list(np.round((timepoint_gaze))*fps)))]=1
        except: # some videos are shorter than the task 
            # timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            # indd = np.array(list(map(int,list(np.round((timepoint_gaze))*fps))))
            # indd = list(indd[indd<np.shape(gaze_angle_speed)[0]])
            # timeseries_gaze[indd] = 1
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        # socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)
        socialgaze_prob = timeseries_gaze
        # 

        #
        # get the self sync pull 
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_syncpull = tpoint_pull2_to_pull1+session_start_time
        elif ianimal == 1:
            timepoint_syncpull = tpoint_pull1_to_pull2+session_start_time
        #
        try:
            timeseries_syncpull = np.zeros((totalsess_nframes,))
            timeseries_syncpull[list(map(int,list(np.round((timepoint_syncpull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_syncpull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_syncpull*fps))))+1,))
            timeseries_syncpull[list(map(int,list(np.round(timepoint_syncpull*fps))))]=1
        # sync_pull_prob = scipy.ndimage.gaussian_filter1d(timeseries_syncpull,gausKernelsize)
        sync_pull_prob = timeseries_syncpull
        # 
        
        #
        # get the gaze lead pull
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_gazepull = tpoint_gaze1_to_pull1+session_start_time
        elif ianimal == 1:
            timepoint_gazepull = tpoint_gaze2_to_pull2+session_start_time
        #
        try:
            timeseries_gazepull = np.zeros((totalsess_nframes,))
            timeseries_gazepull[list(map(int,list(np.round((timepoint_gazepull))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_gazepull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gazepull*fps))))+1,))
            timeseries_gazepull[list(map(int,list(np.round(timepoint_gazepull*fps))))]=1
        # gaze_lead_pull_prob = scipy.ndimage.gaussian_filter1d(timeseries_gazepull,gausKernelsize)
        gaze_lead_pull_prob = timeseries_gazepull
        # 
        
        #
        # get the social attention
        # align to the start of the video recording
        if ianimal == 0:
            timepoint_pullgaze = tpoint_pull2_to_gaze1+session_start_time
        elif ianimal == 1:
            timepoint_pullgaze = tpoint_pull1_to_gaze2+session_start_time
        #
        try:
            timeseries_pullgaze = np.zeros((totalsess_nframes,))
            timeseries_pullgaze[list(map(int,list(np.round((timepoint_pullgaze))*fps)))]=1
        except: # some videos are shorter than the task 
            timeseries_pullgaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_pullgaze*fps))))+1,))
            timeseries_pullgaze[list(map(int,list(np.round(timepoint_pullgaze*fps))))]=1
        # gaze_lead_pull_prob = scipy.ndimage.gaussian_filter1d(timeseries_pullgaze,gausKernelsize)
        social_attention_prob = timeseries_pullgaze
        # 

        # get the forcelevel information
        # align to the start of the video recording
        if ianimal == 0:
            force_all_idate = force1_all_idate
        elif ianimal == 1:
            force_all_idate = force2_all_idate
        blockstarttime_all_idate = blockstarttime_all_idate + session_start_time
        blockendtime_all_idate = blockendtime_all_idate + session_start_time      
        #
        timeseries_forcelevel = np.zeros((totalsess_nframes,))
        nforcetypes = np.shape(force_all_idate)[0]
        for iforcetype in np.arange(0,nforcetypes,1):
            timeseries_forcelevel[int(np.round(blockstarttime_all_idate[iforcetype]*fps)):int(np.round(blockendtime_all_idate[iforcetype]*fps))] = force_all_idate[iforcetype]
        
        leverpull_prob = leverpull_prob[starttimeframe:endtimeframe]
        juice_prob = juice_prob[starttimeframe:endtimeframe]
        socialgaze_prob = socialgaze_prob[starttimeframe:endtimeframe]
        sync_pull_prob = sync_pull_prob[starttimeframe:endtimeframe]
        gaze_lead_pull_prob = gaze_lead_pull_prob[starttimeframe:endtimeframe]
        social_attention_prob = social_attention_prob[starttimeframe:endtimeframe]
        #
        timeseries_forcelevel = timeseries_forcelevel[starttimeframe:endtimeframe]

        # put all the data together in the same order as the con_vars_plot
        if ianimal == 0:
            data_summary[animal1] = [leverpull_prob, socialgaze_prob, juice_prob,
                                     sync_pull_prob, gaze_lead_pull_prob, social_attention_prob, timeseries_forcelevel ]
        elif ianimal == 1:
            data_summary[animal2] = [leverpull_prob, socialgaze_prob, juice_prob,
                                     sync_pull_prob, gaze_lead_pull_prob, social_attention_prob, timeseries_forcelevel ]



    # get the spike related data


    # align the spike time to the start of the video recording
    # only get the spike_time that is within the range of starttime(frame) and endtime(frame)
    spike_time_data_new = spike_time_data + session_start_time*fps

    spike_clusters_unique = np.unique(spike_clusters_data)
    nclusters = np.shape(spike_clusters_unique)[0]

    spiketrain_summary = dict.fromkeys(spike_clusters_unique,[])

    for icluster in np.arange(0,nclusters,1):
        iclusterID = spike_clusters_unique[icluster]

        ind_clusterID = np.isin(spike_clusters_data,iclusterID)

        spike_time_icluster = spike_time_data_new[ind_clusterID]
        spike_time_icluster = np.array(list(map(int,spike_time_icluster)))
        spiketimes, counts = np.unique(spike_time_icluster, return_counts=True)

        nspikes = np.shape(spiketimes)[0]
        spiketrain = np.zeros((1,int(np.max(spike_time_data_new)+1)))[0]

        for ispike in np.arange(0,nspikes,1):
            spiketrain[spiketimes[ispike]] = counts[ispike]

        spiketrain_summary[iclusterID] = spiketrain[starttimeframe:endtimeframe]




    return data_summary, data_summary_names, spiketrain_summary







########################

def neuralGLM_fitting(animal1, animal2, data_summary_names, data_summary, spiketrain_summary, bhvvaris_toGLM, nbootstraps, traintestperc, trig_twin, dospikehist, spikehist_twin, doplots, date_tgt, savefig, save_path, dostrategies, donullshuffle, doforcelevel):

    
    # nbootstraps: how many repreat of the GLM fitting
    # samplesize: the size of the sample, same for pull events and no-pull events

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    
    import random as random
    import seaborn

    import statsmodels.api as sm

    fps = 30 

    nbhvvaris = np.shape(bhvvaris_toGLM)[0]

    # organize the data 
    GLM_bhvtimeseries = []

    for ibhvvari in np.arange(0,nbhvvaris,1):

        bhvvaris = bhvvaris_toGLM[ibhvvari]

        actanimalid = bhvvaris.split(' ')[0]
        bhvvarname = bhvvaris.split(' ')[1]

        if actanimalid == 'self':
            if (animal1 == 'kanga') | (animal2 == 'kanga'):
                actanimal = 'kanga'
            elif (animal1 == 'dodson') | (animal2 == 'dodson'):
                actanimal = 'dodson'
        #         
        elif actanimalid == 'other':
            if (animal1 == 'kanga') | (animal1 == 'dodson'):
                actanimal = animal2
            elif (animal2 == 'kanga') | (animal2 == 'dodson'):
                actanimal = animal1

        #
        bhvtimeseries = data_summary[actanimal][np.where(np.isin(data_summary_names,bhvvarname))[0][0]]
        #
        min_val = np.nanmin(bhvtimeseries)  # Minimum ignoring NaN
        max_val = np.nanmax(bhvtimeseries)  # Maximum ignoring NaN
        # 
        bhvtimeseries = (bhvtimeseries - min_val) / (max_val - min_val)

        # get the force level
        if (animal1 == 'kanga') | (animal2 == 'kanga'):
            selfanimal = 'kanga'
            partneranimal = 'dannon'
        elif (animal1 == 'dodson') | (animal2 == 'dodson'):
            actanimal = 'dodson'
            partneranimal = 'ginger'
        #
        selfforce_timeseries = data_summary[selfanimal][-1]
        partnerforce_timeseries = data_summary[partneranimal][-1]

        # 
        if ibhvvari == 0:
            GLM_bhvtimeseries = bhvtimeseries

        elif ibhvvari == 1:

            if np.shape(bhvtimeseries)[0] > np.shape(GLM_bhvtimeseries)[0]:
                bhvtimeseries = bhvtimeseries[0:np.shape(GLM_bhvtimeseries)[0]]
            elif np.shape(bhvtimeseries)[0] < np.shape(GLM_bhvtimeseries)[0]:
                GLM_bhvtimeseries = GLM_bhvtimeseries[0:np.shape(bhvtimeseries)[0]]

            GLM_bhvtimeseries = np.vstack((GLM_bhvtimeseries,bhvtimeseries))

        else:
            if np.shape(bhvtimeseries)[0] > np.shape(GLM_bhvtimeseries)[1]:
                bhvtimeseries = bhvtimeseries[0:np.shape(GLM_bhvtimeseries)[1]]
            elif np.shape(bhvtimeseries)[0] < np.shape(GLM_bhvtimeseries)[1]:
                GLM_bhvtimeseries = GLM_bhvtimeseries[:,0:np.shape(bhvtimeseries)[0]]

            GLM_bhvtimeseries = np.vstack((GLM_bhvtimeseries,bhvtimeseries))


    # First, prepare the sldiing window data for GLM - for both x and y
    # Then, run GLM based on nbootstraps (number of bootstraps) and traintestperc (percentage of training dataset)
    # for each neuron cluster
    #
    neuron_clusters = list(spiketrain_summary.keys())
    nclusters = np.shape(neuron_clusters)[0]

    try:
        nframesall = np.shape(GLM_bhvtimeseries)[1]
        nbhvtypes = np.shape(GLM_bhvtimeseries)[0]
    except:
        nframesall = np.shape(GLM_bhvtimeseries)[0]
        nbhvtypes = 1

    trig_startframe = int(np.round(trig_twin[0]*fps))
    trig_endframe = int(np.round(trig_twin[1]*fps))
    ntrig_frames = trig_endframe - trig_startframe

    #
    Kernel_coefs_allboots_allcells = dict.fromkeys(neuron_clusters,[])
    Kernel_spikehist_allboots_allcells  = dict.fromkeys(neuron_clusters,[])
    Kernel_selfforce_allboots_allcells = dict.fromkeys(neuron_clusters,[])
    Kernel_partnerforce_allboots_allcells = dict.fromkeys(neuron_clusters,[])

    Kernel_coefs_allboots_allcells_shf = dict.fromkeys(neuron_clusters,[])
    Kernel_spikehist_allboots_allcells_shf  = dict.fromkeys(neuron_clusters,[])
    Kernel_selfforce_allboots_allcells_shf = dict.fromkeys(neuron_clusters,[])
    Kernel_partnerforce_allboots_allcells_shf = dict.fromkeys(neuron_clusters,[])

    # initialize the figure
    if doplots:
        if dospikehist:
            if not doforcelevel:
                fig2, axs2 = plt.subplots(nbhvvaris+1,nclusters)
                fig2.set_figheight(3*(nbhvvaris+1))
                fig2.set_figwidth(4*nclusters)
            elif doforcelevel:
                fig2, axs2 = plt.subplots(nbhvvaris+3,nclusters)
                fig2.set_figheight(3*(nbhvvaris+3))
                fig2.set_figwidth(4*nclusters)
        else: 
            if not doforcelevel:     
                fig2, axs2 = plt.subplots(nbhvvaris,nclusters)
                fig2.set_figheight(3*nbhvvaris)
                fig2.set_figwidth(4*nclusters)
            elif doforcelevel:
                fig2, axs2 = plt.subplots(nbhvvaris+2,nclusters)
                fig2.set_figheight(3*(nbhvvaris+2))
                fig2.set_figwidth(4*nclusters)

    #    
    # for icluster in np.arange(0,1,1):
    for icluster in np.arange(0,nclusters,1):
        iclusterID = neuron_clusters[icluster]

        spiketrain = spiketrain_summary[iclusterID]

        y_all = spiketrain[-trig_startframe:(nframesall-trig_endframe)]

        x_selfforce = selfforce_timeseries[-trig_startframe:(nframesall-trig_endframe)]
        x_partnerforce = partnerforce_timeseries[-trig_startframe:(nframesall-trig_endframe)]

        x_ones = np.ones((np.shape(y_all)[0],1))

        x_bhvall = np.ones((np.shape(y_all)[0],nbhvtypes,ntrig_frames)) * np.nan

        for istamp in np.arange(0,np.shape(y_all)[0],1):
            try:
                x_bhvall[istamp,:,:] = GLM_bhvtimeseries[:,istamp:istamp+ntrig_frames]    
            except:
                x_bhvall[istamp,:,:] = GLM_bhvtimeseries[istamp:istamp+ntrig_frames]  

        #
        if dospikehist:
            # normalize the x variable 
            spiketrain = (spiketrain-np.nanmin(spiketrain))/(np.nanmax(spiketrain)-np.nanmin(spiketrain))

            spikehist_twinframe = spikehist_twin * fps
            x_spikehist = np.ones((np.shape(y_all)[0],spikehist_twinframe))*np.nan
            #
            for istamp in np.arange(0,np.shape(y_all)[0],1):
                x_spikehist[istamp,:] = spiketrain[-trig_startframe-spikehist_twinframe+istamp:-trig_startframe+istamp]

        # 
        if donullshuffle:
            y_all_shf = y_all.copy() 
            random.shuffle(y_all_shf)

        # y_all: spike count train; x_ones: noise term; x_bhvall: the targeted behavioral variables; x_selfforce and x_partnerforce; y_spikehist: spike count history

        # repeat in the bootstraps
        for ibtstrps in np.arange(0,nbootstraps,1):

            nallsamples = np.shape(y_all)[0]

            ntrainsample = int(np.round(traintestperc * nallsamples))

            trainsampleID = random.sample(range(nallsamples), ntrainsample)
            testsampleID = list(np.arange(0,nallsamples,1)[~np.isin(np.arange(0,nallsamples,1),trainsampleID)])

            # training data set
            y_train = y_all[trainsampleID]
            x_selfforce_train = x_selfforce[trainsampleID]
            x_selfforce_train = x_selfforce_train.reshape(-1, 1)  # Convert to 2D
            x_partnerforce_train = x_partnerforce[trainsampleID]
            x_partnerforce_train = x_partnerforce_train.reshape(-1, 1)  # Convert to 2D
            x_ones_train = x_ones[trainsampleID]
            x_bhvall_train = x_bhvall[trainsampleID]
            if dospikehist:
                x_spikehist_train = x_spikehist[trainsampleID]
            #
            if donullshuffle:
                y_train_shf = y_all_shf[trainsampleID]

            #
            for ibhvtype in np.arange(0,nbhvtypes,1):

                if ibhvtype == 0:
                    X_train = x_bhvall_train[:,ibhvtype,:]
                else:
                    X_train = np.hstack((X_train,x_bhvall_train[:,ibhvtype,:]))
            #
            # X_train = np.hstack((X_train,x_ones_train))
            if dospikehist:
                X_train = np.hstack((X_train,x_spikehist_train))
            if doforcelevel:
                X_train = np.hstack((X_train,x_selfforce_train,x_partnerforce_train))

            # remove nan in y and x
            ind_nan = np.isnan(np.sum(X_train,axis=1)) | np.isnan(y_train)
            X_train = X_train[~ind_nan,:]
            y_train = y_train[~ind_nan]

            # Fit a Poisson GLM
            try:
                poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
                poisson_results = poisson_model.fit(method="lbfgs")

                # get the kernel coefficient
                for ibhvtype in np.arange(0,nbhvtypes,1):

                    kernel_ibhv = poisson_results.params[ibhvtype*ntrig_frames:(ibhvtype+1)*ntrig_frames]
                    kernel_ibhv = scipy.ndimage.gaussian_filter1d(kernel_ibhv,10)

                    if ibhvtype == 0:
                        Kernel_coefs = kernel_ibhv
                    else:
                        Kernel_coefs = np.vstack((Kernel_coefs,kernel_ibhv))
            except:
                Kernel_coefs = np.ones((nbhvvaris,ntrig_frames))*np.nan

            if ibtstrps == 0:
                Kernel_coefs_allboots = np.expand_dims(Kernel_coefs, axis=0)
            else:
                Kernel_coefs = np.expand_dims(Kernel_coefs, axis=0)
                Kernel_coefs_allboots = np.vstack((Kernel_coefs_allboots,Kernel_coefs))

            # get the neural history kernel
            if dospikehist:
                try:
                    kernel_spikehist = poisson_results.params[nbhvtypes*ntrig_frames:nbhvtypes*ntrig_frames+spikehist_twinframe]
                    kernel_spikehist = scipy.ndimage.gaussian_filter1d(kernel_spikehist,10)
                except:
                    kernel_spikehist = np.ones((1,spikehist_twinframe))*np.nan
                #
                if ibtstrps == 0:
                    Kernel_spikehist_allboots = kernel_spikehist
                else:
                    Kernel_spikehist_allboots = np.vstack((Kernel_spikehist_allboots,kernel_spikehist))

            # get the kernel for self force and partner force
            if doforcelevel:
                try:
                    kernel_selfforce = poisson_results.params[-2]
                    kernel_partnerforce = poisson_results.params[-1]     
                except:
                    kernel_selfforce = np.nan
                    kernel_partnerforce = np.nan
                #
                if ibtstrps == 0:
                    Kernel_selfforce_allboots = kernel_selfforce
                    Kernel_partnerforce_allboots = kernel_partnerforce
                else:
                    Kernel_selfforce_allboots = np.vstack((Kernel_selfforce_allboots,kernel_selfforce)) 
                    Kernel_partnerforce_allboots = np.vstack((Kernel_partnerforce_allboots,kernel_partnerforce)) 


            # fit Poisson GLM for shuffle data
            if donullshuffle:
                # Fit a Poisson GLM
                try:
                    poisson_model = sm.GLM(y_train_shf, X_train, family=sm.families.Poisson())
                    poisson_results = poisson_model.fit(method="lbfgs")

                    # get the kernel coefficient
                    for ibhvtype in np.arange(0,nbhvtypes,1):

                        kernel_ibhv = poisson_results.params[ibhvtype*ntrig_frames:(ibhvtype+1)*ntrig_frames]
                        kernel_ibhv = scipy.ndimage.gaussian_filter1d(kernel_ibhv,10)

                        if ibhvtype == 0:
                            Kernel_coefs = kernel_ibhv
                        else:
                            Kernel_coefs = np.vstack((Kernel_coefs,kernel_ibhv))
                except:
                    Kernel_coefs = np.ones((nbhvvaris,ntrig_frames))*np.nan

                if ibtstrps == 0:
                    Kernel_coefs_allboots_shf = np.expand_dims(Kernel_coefs, axis=0)
                else:
                    Kernel_coefs = np.expand_dims(Kernel_coefs, axis=0)
                    Kernel_coefs_allboots_shf = np.vstack((Kernel_coefs_allboots_shf,Kernel_coefs))

                # get the neural history kernel
                if dospikehist:
                    try:
                        kernel_spikehist = poisson_results.params[nbhvtypes*ntrig_frames:nbhvtypes*ntrig_frames+spikehist_twinframe]
                        kernel_spikehist = scipy.ndimage.gaussian_filter1d(kernel_spikehist,10)
                    except:
                        kernel_spikehist = np.ones((1,spikehist_twinframe))*np.nan

                    #
                    if ibtstrps == 0:
                        Kernel_spikehist_allboots_shf = kernel_spikehist
                    else:
                        Kernel_spikehist_allboots_shf = np.vstack((Kernel_spikehist_allboots_shf,kernel_spikehist))

                # get the kernel for self force and partner force
                if doforcelevel:
                    try:
                        kernel_selfforce = poisson_results.params[-2]
                        kernel_partnerforce = poisson_results.params[-1]     
                    except:
                        kernel_selfforce = np.nan
                        kernel_partnerforce = np.nan
                    #
                    if ibtstrps == 0:
                        Kernel_selfforce_allboots_shf = kernel_selfforce
                        Kernel_partnerforce_allboots_shf = kernel_partnerforce
                    else:
                        Kernel_selfforce_allboots_shf = np.vstack((Kernel_selfforce_allboots_shf,kernel_selfforce)) 
                        Kernel_partnerforce_allboots_shf = np.vstack((Kernel_partnerforce_allboots_shf,kernel_partnerforce)) 

        # pull the data in the summerizing dataset 
        Kernel_coefs_allboots_allcells[iclusterID] = Kernel_coefs_allboots
        if dospikehist:
            Kernel_spikehist_allboots_allcells[iclusterID] = Kernel_spikehist_allboots 
        if doforcelevel:
            Kernel_selfforce_allboots_allcells[iclusterID] = Kernel_selfforce_allboots
            Kernel_partnerforce_allboots_allcells[iclusterID] = Kernel_partnerforce_allboots
        #
        if donullshuffle:
            Kernel_coefs_allboots_allcells_shf[iclusterID] = Kernel_coefs_allboots_shf
            if dospikehist:
                Kernel_spikehist_allboots_allcells_shf[iclusterID] = Kernel_spikehist_allboots_shf
            if doforcelevel:
                Kernel_selfforce_allboots_allcells_shf[iclusterID] = Kernel_selfforce_allboots_shf
                Kernel_partnerforce_allboots_allcells_shf[iclusterID] = Kernel_partnerforce_allboots_shf




        # plot 
        if doplots:
            #
            for ibhvvari in np.arange(0,nbhvvaris,1):

                xxx_forplot = np.arange(trig_twin[0]*fps, trig_twin[1]*fps, 1)

                Kernel_coefs_forplot = Kernel_coefs_allboots[:,ibhvvari,:]

                mean_trig_trace = np.nanmean(Kernel_coefs_forplot,axis=0)
                std_trig_trace = np.nanstd(Kernel_coefs_forplot,axis=0)
                sem_trig_trace = np.nanstd(Kernel_coefs_forplot,axis=0)/np.sqrt(np.shape(Kernel_coefs_forplot)[0])
                itv95_trig_trace = 1.96*sem_trig_trace

                #
                axs2[ibhvvari,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,
                                                 color='#666666',ecolor='#666666')
                axs2[ibhvvari,icluster].plot([0,0],
                                             [np.nanmin(mean_trig_trace-itv95_trig_trace),np.nanmax(mean_trig_trace+itv95_trig_trace)],
                                             '--k')
                axs2[ibhvvari,icluster].plot([xxx_forplot[0],xxx_forplot[-1]],[0,0], '--k')

                #
                if icluster == 0:
                    axs2[ibhvvari,icluster].set_ylabel(bhvvaris_toGLM[ibhvvari])
                # else:
                #     axs2[ibhvvari,icluster].set_yticklabels([])

                if ibhvvari == nbhvvaris-1:
                    axs2[ibhvvari,icluster].set_xlabel('time (s)')
                    axs2[ibhvvari,icluster].set_xticks(np.arange(trig_twin[0]*fps, trig_twin[1]*fps,60))
                    axs2[ibhvvari,icluster].set_xticklabels(list(map(str,np.arange(trig_twin[0],trig_twin[1],2))))
                else:
                    axs2[ibhvvari,icluster].set_xticklabels([])

                if ibhvvari == 0:
                    axs2[ibhvvari,icluster].set_title('cluster#'+str(iclusterID))


            if dospikehist:

                xxx_forplot = np.arange(-spikehist_twin*fps, 0, 1)

                Kernel_spikehist_forplot = Kernel_spikehist_allboots 

                mean_trig_trace = np.nanmean(Kernel_spikehist_forplot,axis=0)
                std_trig_trace = np.nanstd(Kernel_spikehist_forplot,axis=0)
                sem_trig_trace = np.nanstd(Kernel_spikehist_forplot,axis=0)/np.sqrt(np.shape(Kernel_spikehist_forplot)[0])
                itv95_trig_trace = 1.96*sem_trig_trace

                #
                axs2[nbhvvaris,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,
                                                 color='#666666',ecolor='#666666')
                axs2[nbhvvaris,icluster].plot([0,0],
                                             [np.nanmin(mean_trig_trace-itv95_trig_trace),np.nanmax(mean_trig_trace+itv95_trig_trace)],
                                             '--k')
                axs2[nbhvvaris,icluster].plot([xxx_forplot[0],xxx_forplot[-1]],[0,0], '--k')

                axs2[nbhvvaris,icluster].set_xlabel('time (s)')
                if icluster == 0:
                    axs2[nbhvvaris,icluster].set_ylabel('spike history')
                axs2[nbhvvaris,icluster].set_xticks(np.arange(-spikehist_twin*fps, 0,60))
                axs2[nbhvvaris,icluster].set_xticklabels(list(map(str,np.arange(-spikehist_twin,0,2))))

            if doforcelevel:
                seaborn.violinplot(ax = axs2[-2,icluster],data = Kernel_selfforce_allboots, 
                                   position = 1, color='#666666')
                if icluster == 0:
                    axs2[-2,icluster].set_ylabel('self force level')
                axs2[-2,icluster].set_xlim(0,3)
                #
                seaborn.violinplot(ax = axs2[-1,icluster],data = Kernel_partnerforce_allboots, 
                                   position = 1, color='#666666')
                if icluster == 0:
                    axs2[-1,icluster].set_ylabel('partner force level')
                axs2[-1,icluster].set_xlim(0,3)


            # 
            # plot the shuffled data
            if donullshuffle:
                for ibhvvari in np.arange(0,nbhvvaris,1):

                    xxx_forplot = np.arange(trig_twin[0]*fps, trig_twin[1]*fps, 1)

                    Kernel_coefs_forplot = Kernel_coefs_allboots_shf[:,ibhvvari,:]

                    mean_trig_trace = np.nanmean(Kernel_coefs_forplot,axis=0)
                    std_trig_trace = np.nanstd(Kernel_coefs_forplot,axis=0)
                    sem_trig_trace = np.nanstd(Kernel_coefs_forplot,axis=0)/np.sqrt(np.shape(Kernel_coefs_forplot)[0])
                    itv95_trig_trace = 1.96*sem_trig_trace

                    #
                    axs2[ibhvvari,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,
                                                     color='#e3e3e3',ecolor='#e3e3e3')


                if dospikehist:

                    xxx_forplot = np.arange(-spikehist_twin*fps, 0, 1)

                    Kernel_spikehist_forplot = Kernel_spikehist_allboots_shf 

                    mean_trig_trace = np.nanmean(Kernel_spikehist_forplot,axis=0)
                    std_trig_trace = np.nanstd(Kernel_spikehist_forplot,axis=0)
                    sem_trig_trace = np.nanstd(Kernel_spikehist_forplot,axis=0)/np.sqrt(np.shape(Kernel_spikehist_forplot)[0])
                    itv95_trig_trace = 1.96*sem_trig_trace

                    #
                    axs2[nbhvvaris,icluster].errorbar(xxx_forplot,mean_trig_trace,yerr=itv95_trig_trace,
                                                     color='#e3e3e3',ecolor='#e3e3e3')

                if doforcelevel:
                    seaborn.violinplot(ax = axs2[-2,icluster],data = Kernel_selfforce_allboots_shf, 
                                       position = 2, color='#e3e3e3')
                    #
                    seaborn.violinplot(ax = axs2[-1,icluster],data = Kernel_partnerforce_allboots_shf, 
                                       position = 2, color='#e3e3e3')





    #
    if savefig:
        if dostrategies:
            fig2.savefig(save_path+"/"+date_tgt+'_GLMfitting_meanKernel_strategyVaris.pdf') 
        else:
            fig2.savefig(save_path+"/"+date_tgt+'_GLMfitting_meanKernel.pdf')               
               
                    
    return Kernel_coefs_allboots_allcells, Kernel_spikehist_allboots_allcells, Kernel_selfforce_allboots_allcells, Kernel_partnerforce_allboots_allcells, Kernel_coefs_allboots_allcells_shf, Kernel_spikehist_allboots_allcells_shf, Kernel_selfforce_allboots_allcells_shf, Kernel_partnerforce_allboots_allcells_shf





 













