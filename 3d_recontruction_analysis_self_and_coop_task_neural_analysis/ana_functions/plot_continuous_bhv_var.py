#  function - plot behavioral events; save the pull-triggered events

def plot_continuous_bhv_var(date_tgt,savefig, animal1, animal2, session_start_time, min_length, succpulls_ornot, time_point_pull1, time_point_pull2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles, output_key_locations):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30 

    nanimals = np.shape(animalnames_videotrack)[0]

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist','othergaze_self_angle','mass_move_speed','gaze_angle_speed']
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m','#458B74','#FFC710','#FF1493']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)','degree','pixel/s','degree/s']


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
        trig_twin = [-6,6] # time window to plot the pull triggered events; unit: s
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
        xxx_time = np.arange(0,min_length,1)/fps

        gaze_other_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name]
        gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,30)  
        gaze_tube_angle = output_allangles['selftube_eye_angle_all_Anipose'][animal_name]
        gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,30)  
        gaze_lever_angle = output_allangles['selflever_eye_angle_all_Anipose'][animal_name]
        gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,30)  

        othergaze_self_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name_other]
        othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,30)  


        a = output_key_locations['facemass_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,30)  

        a = output_key_locations['tube_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,30)  

        a = output_key_locations['lever_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,30)  

        a = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,30)  

        a = np.array(output_allvectors['eye_direction_Anipose'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        gaze_angle_speed = np.empty((1,nframes,))
        gaze_angle_speed[:] = np.nan
        gaze_angle_speed = gaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            gaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,30)  


        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist,othergaze_self_angle,mass_move_speed,gaze_angle_speed]

        for iplot in np.arange(0,nconvarplots,1):
            axs[iplot].plot(xxx_time,data_summary[iplot],'-',color = clrs_plot[iplot])
            axs[iplot].set_xlim(0,min_length/fps)
            axs[iplot].set_xlabel('')
            axs[iplot].set_xticklabels('')
            axs[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs[iplot].set_title(animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs[iplot].set_title(animal2+' '+con_vars_plot[iplot])

            if iplot == nconvarplots-1:
                axs[iplot].set_xlabel('time(s)', fontsize = 14)
                axs[iplot].set_xticks(np.arange(0,min_length,3000)/fps) 
                axs[iplot].set_xticklabels(list(map(str,np.arange(0,min_length,3000)/fps)))

            # plot the pull time stamp
            if ianimal == 0:
                timepoint_pull = time_point_pull1
            elif ianimal == 1:
                timepoint_pull = time_point_pull2
            npulls = np.shape(timepoint_pull)[0]

            pull_trigevent_data = []
            pull_trigevent_data_succtrial = []
            pull_trigevent_data_errtrial = []

            for ipull in np.arange(0,npulls,1):
                # timestemp_ipull = np.round((np.array(timepoint_pull)[ipull]+session_start_time))
                timestemp_ipull = (np.array(timepoint_pull)[ipull]+session_start_time)
                yrange = [np.floor(np.nanmin(data_summary[iplot])),np.ceil(np.nanmax(data_summary[iplot]))]
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
            	axs2[iplot].plot(xxx_trigevent,np.nanmean(pull_trigevent_data,axis=0),'k')
            except:
                axs2[iplot].plot(xxx_trigevent,pull_trigevent_data,'k')
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


    return pull_trig_events_summary, pull_trig_events_succtrial_summary, pull_trig_events_errtrial_summary
