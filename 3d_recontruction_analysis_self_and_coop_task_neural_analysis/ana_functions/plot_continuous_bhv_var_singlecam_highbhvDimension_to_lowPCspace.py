#  function - plot behavioral events; save the pull-triggered events

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import string
import warnings
import pickle   

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_continuous_bhv_var_singlecam_highbhvDimension_to_lowPCspace(date_tgt, aligntwins, savefig, animal1, animal2, session_start_time, min_length, succpulls_ornot, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations):


     #
    self_vars_toPCA_names = ['gaze_other_angle', 'gaze_tube_angle', 'gaze_lever_angle', 'animal_animal_dist',
                            'animal_tube_dist', 'animal_lever_dist', 'mass_move_speed', 'gaze_angle_speed',]
    
    other_vars_toPCA_names = ['othergaze_self_angle',  'othergaze_othertube_angle', 'othergaze_otherlever_angle',
                              'animal_animal_dist', 'otherani_otherlever_dist', 'otherani_othertube_dist',
                              'other_mass_move_speed', 'othergaze_angle_speed']
    
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



    con_vars_plot = ['gaze_other_angle', 'gaze_tube_angle', 'gaze_lever_angle',
                     'animal_animal_dist', 'animal_tube_dist', 'animal_lever_dist', 
                     'mass_move_speed', 'gaze_angle_speed',
                     'othergaze_self_angle', 'othergaze_othertube_angle', 'othergaze_otherlever_angle', 
                     'otherani_otherlever_dist', 'otherani_othertube_dist',
                     'other_mass_move_speed', 'othergaze_angle_speed',
                     'socialgaze_prob','othergaze_prob',
                     'selfpull_prob', 'otherpull_prob',
                     'self_PC1','other_PC1' ]
    
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493','#FF1483','#A9A9A9','#8B4513','#8b4613','#8b4713','#8b4813','#8b4913','#8B4513','#8b4613','#8b4713','#8b4813','#8b4913','#8B4513','#8b4613','#8b4713','#8b4813','#8b4913']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)','degree','pixel/s','pixel/s','degree/s','dist(a.u.)','dist(a.u.)','','','','','','','','','','','','']


    pull_trig_events_summary = {}
    pull_trig_events_succtrial_summary = {}
    pull_trig_events_errtrial_summary = {}


    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

        fig, axs = plt.subplots(nconvarplots,1)
        fig.set_figheight(2*nconvarplots)
        fig.set_figwidth(20)

        # plot the pull triggered continuous variables
        trig_twin = [-aligntwins,aligntwins] # time window to plot the pull triggered events; unit: s
        fig2, axs2 = plt.subplots(nconvarplots,1)
        fig2.set_figheight(2*nconvarplots)
        fig2.set_figwidth(6)

        # plot the pull triggered continuous variables, successful pulls
        fig3, axs3 = plt.subplots(nconvarplots,1)
        fig3.set_figheight(2*nconvarplots)
        fig3.set_figwidth(6)

        # plot the pull triggered continuous variables, failed pulls
        fig4, axs4 = plt.subplots(nconvarplots,1)
        fig4.set_figheight(2*nconvarplots)
        fig4.set_figwidth(6)


        # get the variables
        # 'gaze_other_angle'
        gaze_other_angle = output_allangles['other_eye_angle_all_merge'][animal_name]
        gaze_other_angle = np.array(gaze_other_angle)
        # fill NaNs
        nans = np.isnan(gaze_other_angle)
        if np.any(~nans):
            gaze_other_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_other_angle[~nans])
        #
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,gausKernelsize)  # smooth the curve, use 30 before, change to 3 

        # 'gaze_tube_angle'
        gaze_tube_angle = output_allangles['tube_eye_angle_all_merge'][animal_name]
        gaze_tube_angle = np.array(gaze_tube_angle)
        # fill NaNs
        nans = np.isnan(gaze_tube_angle)
        if np.any(~nans):
            gaze_tube_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_tube_angle[~nans])
        #
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,gausKernelsize)  

        # 'gaze_lever_angle'
        gaze_lever_angle = output_allangles['lever_eye_angle_all_merge'][animal_name]
        gaze_lever_angle = np.array(gaze_lever_angle)
        # fill NaNs
        nans = np.isnan(gaze_lever_angle)
        if np.any(~nans):
            gaze_lever_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_lever_angle[~nans])
        #
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,gausKernelsize)  

       # 'othergaze_self_angle'
        othergaze_self_angle = output_allangles['other_eye_angle_all_merge'][animal_name_other]
        othergaze_self_angle = np.array(othergaze_self_angle)
        # fill NaNs
        nans = np.isnan(othergaze_self_angle)
        if np.any(~nans):
            othergaze_self_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), othergaze_self_angle[~nans])
        #
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,gausKernelsize)  
        
        # 'othergaze_othertube_angle'
        othergaze_othertube_angle = output_allangles['tube_eye_angle_all_merge'][animal_name_other]
        othergaze_othertube_angle = np.array(othergaze_othertube_angle)
        # fill NaNs
        nans = np.isnan(othergaze_othertube_angle)
        if np.any(~nans):
            othergaze_othertube_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), othergaze_othertube_angle[~nans])
        #
        othergaze_othertube_angle = scipy.ndimage.gaussian_filter1d(othergaze_othertube_angle,gausKernelsize) 
        
        # 'othergaze_otherlever_angle'
        othergaze_otherlever_angle = output_allangles['lever_eye_angle_all_merge'][animal_name_other]
        othergaze_otherlever_angle = np.array(othergaze_otherlever_angle)
        # fill NaNs
        nans = np.isnan(othergaze_otherlever_angle)
        if np.any(~nans):
            othergaze_otherlever_angle[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), othergaze_otherlever_angle[~nans])
        #
        othergaze_otherlever_angle = scipy.ndimage.gaussian_filter1d(othergaze_otherlever_angle,gausKernelsize) 

        # animal-animal_dist
        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_animal_dist)
        if np.any(~nans):
            animal_animal_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_animal_dist[~nans])
        #
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,gausKernelsize)  

        # 'animal_tube_dist'
        a = output_key_locations['tube_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_tube_dist)
        if np.any(~nans):
            animal_tube_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_tube_dist[~nans])
        #
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,gausKernelsize)  

        # 'animal_lever_dist'
        a = output_key_locations['lever_loc_all_merge'][animal_name].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(animal_lever_dist)
        if np.any(~nans):
            animal_lever_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), animal_lever_dist[~nans])
        #
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,gausKernelsize)  

        # 'mass_move_speed'
        a = output_key_locations['facemass_loc_all_merge'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        # fill NaNs
        nans = np.isnan(mass_move_speed)
        if np.any(~nans):
            mass_move_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), mass_move_speed[~nans])
        #
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,gausKernelsize)  

        # 'other_mass_move_speed'
        a = output_key_locations['facemass_loc_all_merge'][animal_name_other].transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        other_mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        # fill NaNs
        nans = np.isnan(other_mass_move_speed)
        if np.any(~nans):
            other_mass_move_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), other_mass_move_speed[~nans])
        #
        other_mass_move_speed = scipy.ndimage.gaussian_filter1d(other_mass_move_speed,gausKernelsize)

        # 'gaze angle_speed'
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
        # fill NaNs
        nans = np.isnan(gaze_angle_speed)
        if np.any(~nans):
            gaze_angle_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), gaze_angle_speed[~nans])
        #
        gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,gausKernelsize) 
        
        # 'othergaze angle_speed'
        a = np.array(output_allvectors['head_vect_all_merge'][animal_name_other]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        othergaze_angle_speed = np.empty((1,nframes,))
        othergaze_angle_speed[:] = np.nan
        othergaze_angle_speed = othergaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            othergaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        # fill NaNs
        nans = np.isnan(othergaze_angle_speed)
        if np.any(~nans):
            othergaze_angle_speed[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), othergaze_angle_speed[~nans])
        #
        othergaze_angle_speed = scipy.ndimage.gaussian_filter1d(othergaze_angle_speed,gausKernelsize)  

        # 'otheranimal_otherlever_dist'
        a = output_key_locations['lever_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_otherlever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(otherani_otherlever_dist)
        if np.any(~nans):
            otherani_otherlever_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), otherani_otherlever_dist[~nans])
        #
        otherani_otherlever_dist = scipy.ndimage.gaussian_filter1d(otherani_otherlever_dist,gausKernelsize)

        # 'otheranimal_othertube_dist'
        a = output_key_locations['tube_loc_all_merge'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_merge'][animal_name_other].transpose()
        a_min_b = a - b
        otherani_othertube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # fill NaNs
        nans = np.isnan(otherani_othertube_dist)
        if np.any(~nans):
            otherani_othertube_dist[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), otherani_othertube_dist[~nans])
        #
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
        data_summary = [gaze_other_angle, gaze_tube_angle, gaze_lever_angle,
                        animal_animal_dist, animal_tube_dist, animal_lever_dist, 
                        mass_move_speed, gaze_angle_speed,
                        othergaze_self_angle, othergaze_othertube_angle, othergaze_otherlever_angle, 
                        otherani_otherlever_dist, otherani_othertube_dist,
                        other_mass_move_speed, othergaze_angle_speed,
                        socialgaze_prob,othergaze_prob,
                        selfpull_prob, otherpull_prob,]
        
        # add the self PC1
        indices = [con_vars_plot.index(name) for name in self_vars_toPCA_names]
        # 
        # reduce to smaller dimension using PCA; for self animal (animal1)
        allbhvs_a1 = np.array(data_summary)[indices,:]
        data_for_pca = allbhvs_a1.T
        # Normalize the data (Z-score scaling)
        # Each of the 8 variables will now have a mean of 0 and a standard deviation of 1
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_pca)
        # Initialize and run PCA
        # We want to reduce the 8 variables to 3 principal components
        pca = PCA(n_components=3)
        # Fit the model and transform the data
        principal_components = pca.fit_transform(data_scaled)
        principal_components_transposed = principal_components.T
        # Get the explained variance for each component
        explained_variance = pca.explained_variance_ratio_
        #
        PC1_a1 = principal_components_transposed[0,:]
        #
        data_summary.append(PC1_a1)
        
        # add the other PC1
        indices = [con_vars_plot.index(name) for name in other_vars_toPCA_names]
        # 
        # reduce to smaller dimension using PCA; for self animal (animal2)
        allbhvs_a2 = np.array(data_summary)[indices,:]
        data_for_pca = allbhvs_a2.T
        # Normalize the data (Z-score scaling)
        # Each of the 8 variables will now have a mean of 0 and a standard deviation of 1
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_pca)
        # Initialize and run PCA
        # We want to reduce the 8 variables to 3 principal components
        pca = PCA(n_components=3)
        # Fit the model and transform the data
        principal_components = pca.fit_transform(data_scaled)
        principal_components_transposed = principal_components.T
        # Get the explained variance for each component
        explained_variance = pca.explained_variance_ratio_
        #
        PC1_a2 = principal_components_transposed[0,:]
        #
        data_summary.append(PC1_a2)
        

        for iplot in np.arange(0,nconvarplots,1):

            # plot the pull time stamp
            if ianimal == 0:
                timepoint_pull = time_point_pull1
            elif ianimal == 1:
                timepoint_pull = time_point_pull2
            npulls = np.shape(timepoint_pull)[0]

            pull_trigevent_data = []
            pull_trigevent_data_succtrial = []
            pull_trigevent_data_errtrial = []

            # 
            xxx_time = np.arange(0,np.shape(data_summary[iplot])[0],1)/fps

            # only plot the active meaning period
            xxx_time_range = [np.max([xxx_time[0],np.array(timepoint_pull)[0]+session_start_time-5]),
                              np.min([xxx_time[-1],np.array(timepoint_pull)[-1]+session_start_time+5])]
            ind_time_range = (xxx_time >= xxx_time_range[0]) & (xxx_time <=xxx_time_range[1])

            axs[iplot].plot(xxx_time[ind_time_range],data_summary[iplot][ind_time_range],'-',color = clrs_plot[iplot])
            # axs[iplot].set_xlim(0,min_length/fps)
            axs[iplot].set_xlim(xxx_time_range[0],xxx_time_range[1])
            axs[iplot].set_xlabel('')
            axs[iplot].set_xticklabels('')
            axs[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs[iplot].set_title(animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs[iplot].set_title(animal2+' '+con_vars_plot[iplot])

            if iplot == nconvarplots-1:
                axs[iplot].set_xlabel('time(s)', fontsize = 14)
                axs[iplot].set_xticks(np.arange(xxx_time_range[0],xxx_time_range[1],100)) 
                axs[iplot].set_xticklabels(list(map(str,np.arange(xxx_time_range[0]-session_start_time,
                                                                  xxx_time_range[1]-session_start_time,100))))


            #
            for ipull in np.arange(0,npulls,1):
                # timestemp_ipull = np.round((np.array(timepoint_pull)[ipull]+session_start_time))
                timestemp_ipull = (np.array(timepoint_pull)[ipull]+session_start_time)
                # yrange = [np.floor(np.nanmin(data_summary[iplot])),np.ceil(np.nanmax(data_summary[iplot]))]
                yrange = [np.floor(np.nanmin(data_summary[iplot][ind_time_range])),
                          (np.nanmax(data_summary[iplot][ind_time_range]))*1.05]
                axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'k-')

                # plot pull triggered events
                frame_win_ipull = (timestemp_ipull+trig_twin)*fps
                xxx_trigevent = np.arange(trig_twin[0],trig_twin[1],1/fps) 

                try:
                    axs2[iplot].plot(xxx_trigevent,data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iplot])
                    #
                    pull_trigevent_data.append(data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                except:
                    pull_trigevent_data.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])

                # separate trace for the successful pulls
                if 0:
                    if succpulls_ornot[ianimal][ipull]==1:
                        try:
                            axs3[iplot].plot(xxx_trigevent,data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iplot])
                            #
                            pull_trigevent_data_succtrial.append(data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                        except:
                            pull_trigevent_data_succtrial.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])

                    # separate trace for the failed pulls
                    if succpulls_ornot[ianimal][ipull]==0:
                        try:
                            axs4[iplot].plot(xxx_trigevent,data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iplot])
                            #
                            pull_trigevent_data_errtrial.append(data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                        except:
                            pull_trigevent_data_errtrial.append(np.full((1,np.shape(xxx_trigevent)[0]),np.nan)[0])



            # save data into the summary data set
            if ianimal == 0:
                pull_trig_events_summary[(animal1,con_vars_plot[iplot])] = pull_trigevent_data
                pull_trig_events_succtrial_summary[(animal1,con_vars_plot[iplot])] = pull_trigevent_data_succtrial
                pull_trig_events_errtrial_summary[(animal1,con_vars_plot[iplot])] = pull_trigevent_data_errtrial
            elif ianimal == 1:
                pull_trig_events_summary[(animal2,con_vars_plot[iplot])] = pull_trigevent_data
                pull_trig_events_succtrial_summary[(animal2,con_vars_plot[iplot])] = pull_trigevent_data_succtrial
                pull_trig_events_errtrial_summary[(animal2,con_vars_plot[iplot])] = pull_trigevent_data_errtrial

            # plot settings for axs2        
            axs2[iplot].plot([0,0],yrange,'k-')
            try: 
                # axs2[iplot].plot(xxx_trigevent,np.nanmean(pull_trigevent_data,axis=0),'k')
                 # Compute mean and 95% confidence interval
                mean_data = np.nanmean(pull_trigevent_data, axis=0)
                n_samples = np.sum(~np.isnan(pull_trigevent_data), axis=0)
                sem_data = np.nanstd(pull_trigevent_data, axis=0) / np.sqrt(n_samples)
                ci95 = 1.96 * sem_data
                # Plot mean
                axs2[iplot].plot(xxx_trigevent, mean_data, color=clrs_plot[iplot])
                # Plot shaded 95% confidence interval
                axs2[iplot].fill_between(xxx_trigevent, mean_data - ci95, mean_data + ci95,
                                         color=clrs_plot[iplot], alpha=0.3)
            except:
                # axs2[iplot].plot(xxx_trigevent,pull_trigevent_data,'k')
                axs2[iplot].plot(xxx_trigevent,pull_trigevent_data,color = clrs_plot[iplot])
            #
            axs2[iplot].set_xlim(trig_twin)
            axs2[iplot].set_xlabel('')
            axs2[iplot].set_xticklabels('')
            axs2[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs2[iplot].set_title('all pulls '+animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs2[iplot].set_title('all pulls '+animal2+' '+con_vars_plot[iplot])
            #
            if iplot == nconvarplots-1:
                axs2[iplot].set_xlabel('time(s)', fontsize = 14)
                axs2[iplot].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs2[iplot].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            # plot settings for axs3        
            axs3[iplot].plot([0,0],yrange,'k-')
            try:
                axs3[iplot].plot(xxx_trigevent,np.nanmean(pull_trigevent_data_succtrial,axis=0),'k')
            except:
                try:
                    axs3[iplot].plot(xxx_trigevent,pull_trigevent_data_succtrial,'k')
                except:
                    pass
            #
            axs3[iplot].set_xlim(trig_twin)
            axs3[iplot].set_xlabel('')
            axs3[iplot].set_xticklabels('')
            axs3[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs3[iplot].set_title('successful pulls '+animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs3[iplot].set_title('successful pulls '+animal2+' '+con_vars_plot[iplot])
            #
            if iplot == nconvarplots-1:
                axs3[iplot].set_xlabel('time(s)', fontsize = 14)
                axs3[iplot].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs3[iplot].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            # plot settings for axs4        
            axs4[iplot].plot([0,0],yrange,'k-')
            try:
                axs4[iplot].plot(xxx_trigevent,np.nanmean(pull_trigevent_data_errtrial,axis=0),'k')
            except:
                try:
                    axs4[iplot].plot(xxx_trigevent,pull_trigevent_data_errtrial,'k')
                except:
                    pass
            #
            axs4[iplot].set_xlim(trig_twin)
            axs4[iplot].set_xlabel('')
            axs4[iplot].set_xticklabels('')
            axs4[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs4[iplot].set_title('failed pulls '+animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs4[iplot].set_title('failed pulls '+animal2+' '+con_vars_plot[iplot])
            #
            if iplot == nconvarplots-1:
                axs4[iplot].set_xlabel('time(s)', fontsize = 14)
                axs4[iplot].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs4[iplot].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            if savefig:
                if ianimal == 0:
                    fig.savefig(date_tgt+'_'+animal1+"_continuous_bhv_variables.pdf")
                    fig2.savefig(date_tgt+'_'+animal1+"_pull_triggered_bhv_variables.pdf")
                    fig3.savefig(date_tgt+'_'+animal1+"_successfulpull_triggered_bhv_variables.pdf")
                    fig4.savefig(date_tgt+'_'+animal1+"_failedpull_triggered_bhv_variables.pdf")
                elif ianimal == 1:
                    fig.savefig(date_tgt+'_'+animal2+"_continuous_bhv_variables.pdf")
                    fig2.savefig(date_tgt+'_'+animal2+"_pull_triggered_bhv_variables.pdf")
                    fig3.savefig(date_tgt+'_'+animal2+"_successfulpull_triggered_bhv_variables.pdf")
                    fig4.savefig(date_tgt+'_'+animal2+"_failedpull_triggered_bhv_variables.pdf")

    plt.close('all')


    return pull_trig_events_summary, pull_trig_events_succtrial_summary, pull_trig_events_errtrial_summary
