#  function - plot behavioral events

def plot_continuous_bhv_var(date_tgt,animal1, animal2, session_start_time, min_length, time_point_pull1, time_point_pull2, animalnames_videotrack, output_look_ornot, output_allvectors, output_allangles,output_key_locations):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30 

    nanimals = np.shape(animalnames_videotrack)[0]

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist']
    nconvarplots = np.shape(con_vars_plot)[0]

    clrs_plot = ['r','y','g','b','c','m']
    yaxis_labels = ['degree','degree','degree','dist(a.u.)','dist(a.u.)','dist(a.u.)']

    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

        fig, axs = plt.subplots(nconvarplots,1)
        fig.set_figheight(12)
        fig.set_figwidth(20)

        # plot the pull triggered continuous variables
        fig2, axs2 = plt.subplots(nconvarplots,1)
        fig2.set_figheight(15)
        fig2.set_figwidth(6)
        trig_twin = [-6,6] # time window to plot the pull triggered events; unit: s


        # get the variables
        xxx_time = np.arange(0,min_length,1)/fps

        gaze_other_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name]
        gaze_tube_angle = output_allangles['selftube_eye_angle_all_Anipose'][animal_name]
        gaze_lever_angle = output_allangles['selflever_eye_angle_all_Anipose'][animal_name]

        a = output_key_locations['facemass_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))

        a = output_key_locations['tube_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))

        a = output_key_locations['lever_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))

        # put all the data together in the same order as the con_vars_plot
        data_summary = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist]

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
            for ipull in np.arange(0,npulls,1):
                timestemp_ipull = np.round((np.array(timepoint_pull)[ipull]+session_start_time))
                yrange = [np.floor(np.nanmin(data_summary[iplot])),np.ceil(np.nanmax(data_summary[iplot]))]
                axs[iplot].plot([timestemp_ipull,timestemp_ipull],yrange,'k-')

                # plot pull triggered events
                frame_win_ipull = (timestemp_ipull+trig_twin)*fps
                xxx_trigevent = np.arange(trig_twin[0],trig_twin[1],1/fps)
                try:
                    axs2[iplot].plot(xxx_trigevent,data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])],'-',color = clrs_plot[iplot])
                    pull_trigevent_data.append(data_summary[iplot][int(frame_win_ipull[0]):int(frame_win_ipull[1])])
                except:
                    continue
            axs2[iplot].plot([0,0],yrange,'k-')
            axs2[iplot].plot(xxx_trigevent,np.nanmean(pull_trigevent_data,axis=0),'k')

            axs2[iplot].set_xlim(trig_twin)
            axs2[iplot].set_xlabel('')
            axs2[iplot].set_xticklabels('')
            axs2[iplot].set_ylabel(yaxis_labels[iplot])
            if ianimal == 0:
                axs2[iplot].set_title(animal1+' '+con_vars_plot[iplot])
            elif ianimal == 1:
                axs2[iplot].set_title(animal2+' '+con_vars_plot[iplot])

            if iplot == nconvarplots-1:
                axs2[iplot].set_xlabel('time(s)', fontsize = 14)
                axs2[iplot].set_xticks(np.linspace(trig_twin[0],trig_twin[1],5)) 
                axs2[iplot].set_xticklabels(list(map(str,np.linspace(trig_twin[0],trig_twin[1],5))))

            if ianimal == 0:
                fig.savefig(date_tgt+'_'+animal1+"_continuous_bhv_variables.pdf")
                fig2.savefig(date_tgt+'_'+animal1+"_pull_triggered_bhv_variables.pdf")
            elif ianimal == 1:
                fig.savefig(date_tgt+'_'+animal2+"_continuous_bhv_variables.pdf")
                fig2.savefig(date_tgt+'_'+animal2+"_pull_triggered_bhv_variables.pdf")



