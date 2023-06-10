#  function - make demo videos for the body part tracking after 3d reconrstruct
def tracking_video_3d_demo(bodyparts_locs_3d,animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename,animal2_filename,session_start_time,fps,nframes,video_file):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    import string
    import warnings
    import pickle
    import cv2
    import os   
    import matplotlib.animation as animation

    import warnings
    warnings.filterwarnings("ignore")    

    skeletons = [ ['rightTuft','rightEye'],
                  ['rightTuft','whiteBlaze'],
                  ['leftTuft','leftEye'],
                  ['leftTuft','whiteBlaze'],
                  ['rightEye','whiteBlaze'],
                  ['leftEye','whiteBlaze'],
                  ['rightEye','mouth'],
                  ['leftEye','mouth'],
                  ['leftEye','rightEye']
                ]
    nskeletons = np.shape(skeletons)[0]
    
    colors = ['b','r','k']
    

    # Settings
    clear_frames = True     # Should it clear the figure between each frame?
    fps = 30

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_3d.keys()).iloc[:,0])
    # body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_3d.keys()).iloc[:,1])
    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack    

    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]
    
    for iname in np.arange(0,nanimals,1):
        for ibody in np.arange(0,nbodyparts,1):
            if (iname == 0) & (ibody == 0):
                xxx = np.array(bodyparts_locs_3d[(animal_names_unique[iname],body_parts_unique[ibody])])
            else:
                xxx2 = np.array(bodyparts_locs_3d[(animal_names_unique[iname],body_parts_unique[ibody])])
                xxx = np.concatenate([xxx,xxx2])
    xyz_min = np.nanmin(xxx,axis=0)
    xyz_max = np.nanmax(xxx,axis=0)



    # align the plot with the session start
    iframe_min = int(np.round(session_start_time*fps))
    iframe_max = nframes+iframe_min   


    # set up the figure setting  
    fig = plt.figure(figsize = (15,12))
    #gs=GridSpec(5,3) # 5 rows, 3 columns

    #ax1=fig.add_subplot(gs[0:2,:],projection='3d') # animal tracking frame
    #ax2=fig.add_subplot(gs[3,:]) # animal1 behavioral events
    #ax3=fig.add_subplot(gs[4,:]) # animal2 behavioral events
    ax1 = fig.add_subplot(projection='3d')

    #ax1.set_xlim([-10,10])
    #ax1.set_ylim([-10,10])
    #ax1.set_zlim([70,110])
    ax1.set_xlim([xyz_min[0],xyz_max[0]])
    ax1.set_ylim([xyz_min[1],xyz_max[1]])
    ax1.set_zlim([xyz_min[2],xyz_max[2]])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    #ax2.set_xlim([iframe_min,iframe_max]) 
    #ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    #ax2.set_xticklabels('')
    #ax2.set_ylim([0,1])
    #ax2.set_yticklabels('')
    #ax2.set_xlabel('')
    #ax2.set_ylabel('')
    #ax2.set_title('animal 1 behavioral events')
    
    #ax3.set_xlim([iframe_min,iframe_max])  
    #ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
    #ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
    #ax3.set_ylim([0,1])
    #ax3.set_yticklabels('')
    #ax3.set_xlabel('time (s)')
    #ax3.set_ylabel('')
    #ax3.set_title('animal 2 behavioral events')



    with writer.saving(fig, video_file, 100):
        # for iframe in np.arange(0,nframes,1):    
        for iframe in np.arange(iframe_min,iframe_max,1):
           
            print("printing frame ",str(iframe+1),"/",str(iframe_max))
        
            if clear_frames:
                fig.clear()
                #gs=GridSpec(5,3) # 5 rows, 3 columns

                #ax1=fig.add_subplot(gs[0:2,:],projection='3d') # animal tracking frame
                #ax2=fig.add_subplot(gs[3,:]) # animal1 behavioral events
                #ax3=fig.add_subplot(gs[4,:]) # animal2 behavioral events
                ax1 = fig.add_subplot(projection='3d')

                #ax1.set_xlim([-10,10])
                #ax1.set_ylim([-10,10])
                #ax1.set_zlim([70,110])
                ax1.set_xlim([xyz_min[0],xyz_max[0]])
                ax1.set_ylim([xyz_min[1],xyz_max[1]])
                ax1.set_zlim([xyz_min[2],xyz_max[2]])
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')
                
                #ax2.set_xlim([iframe_min,iframe_max]) 
                #ax2.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                #ax2.set_xticklabels('')
                #ax2.set_ylim([0,1])
                #ax2.set_yticklabels('')
                #ax2.set_xlabel('')
                #ax2.set_ylabel('')
                #ax2.set_title('animal 1 behavioral events')
    
                #ax3.set_xlim([iframe_min,iframe_max])  
                #ax3.set_xticks(np.arange(iframe_min,iframe_max,300)) 
                #ax3.set_xticklabels(list(map(str,np.arange(0/fps,nframes/fps,300/fps))))
                #ax3.set_ylim([0,1])
                #ax3.set_yticklabels('')
                #ax3.set_xlabel('time (s)')
                #ax3.set_ylabel('')
                #ax3.set_title('animal 2 behavioral events')

            
            for ianimal in np.arange(0,nanimals,1):    
                ianimal_name = animal_names_unique[ianimal] 
                # draw body part
                bodypart_loc_iframe = np.zeros((nbodyparts,3))
                #
                for ibdpart in np.arange(0,nbodyparts,1):           

                    ibdpart_name = body_parts_unique[ibdpart]
                    bodypart_loc_iframe[ibdpart,:] = np.array(bodyparts_locs_3d[(ianimal_name,ibdpart_name)])[iframe,:]
                # plot the body parts
                if (ianimal==0): 
                    ax1.plot3D(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1],bodypart_loc_iframe[:,2], '.', color=colors[ianimal],label ='animal1')
                else:
                    ax1.plot3D(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1],bodypart_loc_iframe[:,2], '.', color=colors[ianimal],label ='animal2')
                
                # draw skeleton                
                for iskel in np.arange(0,nskeletons,1):
                    try:
                        iskeleton_name = skeletons[iskel]
                        skelbody12_loc_iframe = np.zeros((2,3))
                        #
                        skel_body1_name = iskeleton_name[0]
                        skel_body2_name = iskeleton_name[1]
                        #
                        skelbody12_loc_iframe[0,:] = np.array(bodyparts_locs_3d[(ianimal_name,skel_body1_name)])[iframe,:]
                        skelbody12_loc_iframe[1,:] = np.array(bodyparts_locs_3d[(ianimal_name,skel_body2_name)])[iframe,:]
                        # plot one skeleton
                        ax1.plot3D(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],skelbody12_loc_iframe[:,2],'-',color=colors[ianimal])
                    except:
                        continue

                
           
                 

                
                ax1.legend(loc='upper right')
                                          
                    

            writer.grab_frame()            
