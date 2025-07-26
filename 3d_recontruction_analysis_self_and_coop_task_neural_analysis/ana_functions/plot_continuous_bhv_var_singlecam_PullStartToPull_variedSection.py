#  function - plot behavioral events; save the pull-triggered events - from pull start to pull action

def plot_continuous_bhv_var_singlecam_PullStartToPull_variedSection(pull1_rt, pull2_rt, animal1, animal2, session_start_time, min_length, succpulls_ornot, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30 

    gausKernelsize = 4 # 15

    gaze_thresold = 0.2 # min length threshold to define if a gaze is real gaze or noise, in the unit of second
    
    nanimals = np.shape(animalnames_videotrack)[0]
    
    # prepare some time stamp data
    # merge oneway gaze and mutual gaze # note!! the time stamps are already aligned to the start of session (instead of the start of video recording)
    oneway_gaze1 = np.sort(np.hstack((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.hstack((oneway_gaze2,mutual_gaze2)))
    
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
    
    

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist','othergaze_self_angle','mass_move_speed', 'other_mass_move_speed', 'gaze_angle_speed','otherani_otherlever_dist','otherani_othertube_dist','socialgaze_prob','othergaze_prob','otherpull_prob', 'selfpull_prob']
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#FF1483','#A9A9A9','#8B4513','#8b4613','#8b4713','#8b4813','#8b4913']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)','degree','pixel/s','pixel/s','degree/s','dist(a.u.)','dist(a.u.)','','','','']


    pull_trig_events_summary = {}
    

    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]


        # get the variables
        gaze_other_angle = output_allangles['other_eye_angle_all_merge'][animal_name]
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  # smooth the curve, use 30 before, change to 3 

        gaze_tube_angle = output_allangles['tube_eye_angle_all_merge'][animal_name]
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  

        gaze_lever_angle = output_allangles['lever_eye_angle_all_merge'][animal_name]
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

        othergaze_self_angle = output_allangles['other_eye_angle_all_merge'][animal_name_other]
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        a = output_key_locations['tube_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        a = output_key_locations['lever_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,gausKernelsize)  

        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        other_mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        other_mass_move_speed = scipy.ndimage.gaussian_filter1d(other_mass_move_speed,gausKernelsize)

        a = np.array(output_allvectors['head_vect_all_merge'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        gaze_angle_speed = np.empty((1,nframes,))
        gaze_angle_speed[:] = np.nan
        gaze_angle_speed = gaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            gaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,gausKernelsize)  

        a = output_key_locations['lever_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_otherlever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        otherani_otherlever_dist = scipy.ndimage.gaussian_filter1d(otherani_otherlever_dist,gausKernelsize)

        a = output_key_locations['tube_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_othertube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        otherani_othertube_dist = scipy.ndimage.gaussian_filter1d(otherani_othertube_dist,gausKernelsize)

        #
        # get the self social gaze time series
        # align to the start of the video recording
        # self social gaze
        if ianimal == 0:
            timepoint_gaze = animal1_gaze+session_start_time
        elif ianimal == 1:
            timepoint_gaze = animal2_gaze+session_start_time
        #
        try:
            timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        socialgaze_prob = timeseries_gaze
        socialgaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)
        
        #
        # get the other social gaze time series
        # align to the start of the video recording
        # other social gaze
        if ianimal == 0:
            timepoint_gaze = animal2_gaze+session_start_time
        elif ianimal == 1:
            timepoint_gaze = animal1_gaze+session_start_time
        #
        try:
            timeseries_gaze = np.zeros(np.shape(gaze_angle_speed))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_gaze = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_gaze*fps))))+1,))
            timeseries_gaze[list(map(int,list(np.round(timepoint_gaze*fps))))]=1
        othergaze_prob = timeseries_gaze
        othergaze_prob = scipy.ndimage.gaussian_filter1d(timeseries_gaze,gausKernelsize)

        #
        # get the other pull time series
        # align to the start of the video recording
        # other pull
        if ianimal == 0:
            timepoint_otherpull = time_point_pull2 + session_start_time
        elif ianimal == 1:
            timepoint_otherpull = time_point_pull1 + session_start_time
        #
        try:
            timeseries_otherpull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_otherpull[list(map(int,list(np.round(timepoint_otherpull*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_otherpull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_otherpull*fps))))+1,))
            timeseries_otherpull[list(map(int,list(np.round(timepoint_otherpull*fps))))]=1
        otherpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_otherpull,1)

        #
        # get the self pull time series for comparison
        # align to the start of the video recording
        # self pull
        if ianimal == 0:
            timepoint_selfpull = time_point_pull1 + session_start_time
        elif ianimal == 1:
            timepoint_selfpull = time_point_pull2 + session_start_time
        #
        try:
            timeseries_selfpull = np.zeros(np.shape(gaze_angle_speed))
            timeseries_selfpull[list(map(int,list(np.round(timepoint_selfpull*fps))))]=1
        except: # some videos are shorter than the task 
            timeseries_selfpull = np.zeros((int(np.ceil(np.nanmax(np.round(timepoint_selfpull*fps))))+1,))
            timeseries_selfpull[list(map(int,list(np.round(timepoint_selfpull*fps))))]=1
        selfpull_prob = scipy.ndimage.gaussian_filter1d(timeseries_selfpull,1)


        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle, animal_animal_dist, animal_tube_dist, animal_lever_dist, othergaze_self_angle, mass_move_speed, other_mass_move_speed, gaze_angle_speed, otherani_otherlever_dist, otherani_othertube_dist, socialgaze_prob, othergaze_prob, otherpull_prob, selfpull_prob]

        for iplot in np.arange(0,nconvarplots,1):
            
            # fill NaNs
            nans = np.isnan(data_summary[iplot])
            if np.any(~nans):
                data_summary[iplot][nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data_summary[iplot][~nans])
            
            xxx_time = np.arange(0,np.shape(data_summary[iplot])[0],1)/fps
            
            # plot the pull time stamp
            if ianimal == 0:
                timepoint_pull = time_point_pull1
                pull_rt = pull1_rt
            elif ianimal == 1:
                timepoint_pull = time_point_pull2
                pull_rt = pull2_rt
            npulls = np.shape(timepoint_pull)[0]

            pull_trigevent_data = []


            for ipull in np.arange(0,npulls,1):
                
                # organize the pull triggered continuous variables with flexible length
                # from the pull onset to the pull action
                trig_twin = [0,pull_rt[ipull]] # time window to plot the pull triggered events; unit: s
                
                # timestemp_ipull = np.round((np.array(timepoint_pull)[ipull]+session_start_time))
                timestemp_ipull = (np.array(timepoint_pull)[ipull]+session_start_time)

                # plot pull triggered events
                frame_win_ipull = (timestemp_ipull+trig_twin)*fps
                xxx_trigevent = np.arange(trig_twin[0],trig_twin[1],1/fps) 

                try:
                    pull_trigevent_data.append(data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                except:
                    pull_trigevent_data.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])
                    
                       
            # save data into the summary data set
            if ianimal == 0:
                pull_trig_events_summary[(animal1,con_vars_plot[iplot])] = pull_trigevent_data
               
            elif ianimal == 1:
                pull_trig_events_summary[(animal2,con_vars_plot[iplot])] = pull_trigevent_data
                

                
           


    return pull_trig_events_summary
