#  function - make demo videos for the body part tracking based on single camera, also show the important axes
def tracking_video_singlecam_demo(bodyparts_locs_camN,output_look_ornot,output_allvectors,output_allangles,lever_loc_both, tube_loc_both,animal1,animal2,animalnames_videotrack,bodypartnames_videotrack,date_tgt,nframes):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle
   
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
    
    import matplotlib.animation as animation

    # Settings
    video_file = "example_videos_singlecam_demo/"+date_tgt+"_cam2only_tracking_demo.mp4"
    clear_frames = True     # Should it clear the figure between each frame?
    fps = 30

    # Output video writer
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # animal_names_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,0])
    # body_parts_unique = pd.unique(pd.DataFrame(bodyparts_locs_camN.keys()).iloc[:,1])
    animal_names_unique = animalnames_videotrack
    body_parts_unique = bodypartnames_videotrack    

    nanimals = np.shape(animal_names_unique)[0]  
    nbodyparts = np.shape(body_parts_unique)[0]
    
   
    # set up the figure setting  
    fig = plt.figure(figsize = (10,5))
    ax = fig.subplots(1,1)
    ax.set_xlim([0,1920])
    ax.set_ylim([0,1080])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.invert_yaxis()

    with writer.saving(fig, video_file, 100):
        for iframe in np.arange(0,nframes,1):    
            if clear_frames:
                fig.clear()
                ax = fig.subplots(1,1)
                ax.set_xlim([0,1920])
                ax.set_ylim([0,1080])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.invert_yaxis()

            
            for ianimal in np.arange(0,nanimals,1):    
                ianimal_name = animal_names_unique[ianimal] 
                # draw body part
                bodypart_loc_iframe = np.zeros((nbodyparts,2))
                #
                for ibdpart in np.arange(0,nbodyparts,1):           

                    ibdpart_name = body_parts_unique[ibdpart]
                    bodypart_loc_iframe[ibdpart,:] = np.array(bodyparts_locs_camN[(ianimal_name,ibdpart_name)])[iframe,:]
                # plot the body parts
                ax.plot(bodypart_loc_iframe[:,0], bodypart_loc_iframe[:,1], '.', color=colors[ianimal])
                
                # draw skeleton                
                for iskel in np.arange(0,nskeletons,1):
                    try:
                        iskeleton_name = skeletons[iskel]
                        skelbody12_loc_iframe = np.zeros((2,2))
                        #
                        skel_body1_name = iskeleton_name[0]
                        skel_body2_name = iskeleton_name[1]
                        #
                        skelbody12_loc_iframe[0,:] = np.array(bodyparts_locs_camN[(ianimal_name,skel_body1_name)])[iframe,:]
                        skelbody12_loc_iframe[1,:] = np.array(bodyparts_locs_camN[(ianimal_name,skel_body2_name)])[iframe,:]
                        # plot one skeleton
                        ax.plot(skelbody12_loc_iframe[:,0],skelbody12_loc_iframe[:,1],'-',color=colors[ianimal])
                    except:
                        continue

                
                # draw lever and tube location
                ax.plot(lever_loc_both[ianimal_name][0],lever_loc_both[ianimal_name][1],'o',color='g')
                ax.plot(tube_loc_both[ianimal_name][0],tube_loc_both[ianimal_name][1],'o',color='y')
           
                # draw head vector
                rightEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]
                leftEye_loc_iframe = np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]
                meaneye_loc_iframe = np.nanmean(np.vstack([rightEye_loc_iframe,leftEye_loc_iframe]),axis=0)
                head_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['head_vect_all_merge'][ianimal_name])[iframe,:]
                ax.plot([meaneye_loc_iframe[0],head_loc_iframe[0]],[meaneye_loc_iframe[1],head_loc_iframe[1]],'-',color = '0.75')

                # draw other - eye vector            
                other_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['other_eye_vect_all_merge'][ianimal_name])[iframe,:]
                ax.plot([meaneye_loc_iframe[0],other_loc_iframe[0]],[meaneye_loc_iframe[1],other_loc_iframe[1]],'-',color = colors[np.absolute(ianimal*2-1)])

                # draw tube - eye vector
                tube_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['tube_eye_vect_all_merge'][ianimal_name])[iframe,:]
                ax.plot([meaneye_loc_iframe[0],tube_loc_iframe[0]],[meaneye_loc_iframe[1],tube_loc_iframe[1]],'-',color = 'y')

                # draw lever - eye vector
                lever_loc_iframe = meaneye_loc_iframe + 200*np.array(output_allvectors['lever_eye_vect_all_merge'][ianimal_name])[iframe,:]
                ax.plot([meaneye_loc_iframe[0],lever_loc_iframe[0]],[meaneye_loc_iframe[1],lever_loc_iframe[1]],'-',color = 'g')

            writer.grab_frame()            
