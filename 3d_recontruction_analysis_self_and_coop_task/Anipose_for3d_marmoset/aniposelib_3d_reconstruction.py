import numpy as np
import aniposelib
import toml
import pandas as pd
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
import os


## section 1 - cammera calibration
# calibrate the cameras for 3d

do_cali = 0

#vidnames = [['camera-1_new_position_20221107.mp4'],
#            ['camera-2_new_position_20221107.mp4'],
#            ['camera-3_new_position_20221107.mp4']]
vidnames = [['camera-1_new_position_20221109_merge.mp4'],
            ['camera-2_new_position_20221109_merge.mp4'],
            ['camera-3_new_position_20221109_merge.mp4']]

cam_names = ['1', '2', '3']

n_cams = len(vidnames)

board = Checkerboard(8, 6,
                     square_length=1, # here, in mm but any unit works
                     manually_verify=False)

# the videos provided are fisheye, so we need the fisheye option
cgroup = CameraGroup.from_names(cam_names, fisheye=False)

if do_cali: 
    # this will take about 15 minutes (mostly due to detection)
    # it will detect the charuco board in the videos,
    # then calibrate the cameras based on the detections, using iterative bundle adjustment
    cgroup.calibrate_videos(vidnames, board)

    # if you need to save and load
    # example saving and loading for later
    cgroup.dump('calibration.toml')

else:
    ## example of loading calibration from a file
    ## you can also load the provided file if you don't want to wait 15 minutes
    cgroup = CameraGroup.load('calibration.toml')


## section 2 - test on videos


# dates for Eddie and Sparkle
analyzed_dates = [
                  "20221122","20221125","20221128","20221129","20221130","20221202","20221206",
                  "20221207","20221208","20221209","20230126","20230127","20230130","20230201","20230203-1",
                  "20230206","20230207","20230208-1","20230209","20230222","20230223-1","20230227-1",
                  "20230228-1","20230302-1","20230307-2","20230313","20230315","20230316","20230317",
                  "20230321","20230322","20230324","20230327","20230328",
                  "20230330","20230331","20230403","20230404","20230405",
                  "20230406","20230407"
               ]

# analyzed_dates = ["20221128"]

session_start_times = [ 
                            8.00,38.00,1.00,3.00,5.00,9.50,1.00,
                            4.50,4.50,5.00,38.00,166.00,4.20,3.80,3.60,
                            7.50,9.00,7.50,8.50,14.50,7.80,8.00,7.50,
                            8.00,8.00,4.00,123.00,14.00,8.80,
                            7.00,7.50,5.50,11.00,9.00,
                            17.00,4.50,9.30,25.50,20.40,
                            21.30,24.80
                         ]

# session_start_times = [1.00]

animal1 = "Eddie"
animal2 = "Sparkle"

lever_locs_all = {'camera-1':{('dodson'):np.array([645, 600]),('scorch'):np.array([425, 435])},
                  'camera-2':{('dodson'):np.array([1335,715]),('scorch'):np.array([550, 715])},
                  'camera-3':{('dodson'):np.array([1580,440]),('scorch'):np.array([1296,540])}}
tube_locs_all = {'camera-1':{('dodson'):np.array([1350,630]),('scorch'):np.array([555, 345])},
                 'camera-2':{('dodson'):np.array([1550,515]),('scorch'):np.array([350, 515])},
                 'camera-3':{('dodson'):np.array([1470,375]),('scorch'):np.array([805,475])}}

ndates = np.shape(analyzed_dates)[0]

singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_middle_cameraSep1shuffle1_150000"
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth']

nanimals_ana = np.shape(animalnames_videotrack)[0]
nbodyparts_ana = np.shape(bodypartnames_videotrack)[0]

do_videodemos = 1

for idate in np.arange(0,ndates,1):
    date_tgt = analyzed_dates[idate]

    twocamera_videos_cam12 = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1+"_"+animal2+"_camera12/"
    twocamera_videos_cam23 = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_cooperative_task_3d/"+date_tgt+"_"+animal1+"_"+animal2+"_camera23/"

    bodyparts_cam1_cam12 = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_camera-1"+singlecam_ana_type+"_el_filtered.h5"
    bodyparts_cam2_cam12 = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_camera-2"+singlecam_ana_type+"_el_filtered.h5"
    bodyparts_cam3_cam23 = twocamera_videos_cam23+date_tgt+"_"+animal1+"_"+animal2+"_camera-3"+singlecam_ana_type+"_el_filtered.h5"

    bodyparts_3d_cam12_DLC = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_weikang.h5"
    bodyparts_3d_cam23_DLC = twocamera_videos_cam23+date_tgt+"_"+animal1+"_"+animal2+"_weikang.h5"

    #
    add_date_dir = twocamera_videos_cam12 +'/anipose_cam123_3d_h5_files/'+date_tgt+'_'+animal1+'_'+animal2
    bodyparts_3d_anipose_file = add_date_dir+'/'+date_tgt+'_'+animal1+'_'+animal2+'_anipose.h5'
    if os.path.exists(bodyparts_3d_anipose_file):
        do_3dconstruct = 0
    else:
        try:
            os.makedirs(add_date_dir)
            do_3dconstruct = 1
        except:
            do_3dconstruct = 1
    #do_3dconstruct=0


    if do_3dconstruct:   
      
        bodyparts_3d_singleAni_anipose_merge = {}
        for ianimal_ana in np.arange(0,nanimals_ana,1):
            
            animalname_ana = animalnames_videotrack[ianimal_ana]
            print('Anipose 3d triangulate witth camera 1, 2 and 3 for '+animalname_ana)

            ## save the the h5 file separately for each animals and save them in the same folder for future purpose

            # animal 1 - "dodson" & animal 2 - "scorch"

            # single animal h5 files
            bodyparts_cam1_cam12_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-1"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
            bodyparts_cam2_cam12_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-2"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
            bodyparts_cam3_cam23_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-3"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
  
            # cam1 
            bodyparts_cam1_cam12_data = pd.read_hdf(bodyparts_cam1_cam12)
            bodyparts_cam1_cam12_singleAni_data = {}
            bodyparts_cam1_cam12_singleAni_data[singlecam_ana_type]=bodyparts_cam1_cam12_data.loc[:,(singlecam_ana_type,animalname_ana)]
            bodyparts_cam1_cam12_singleAni_data=pd.concat(bodyparts_cam1_cam12_singleAni_data, axis=1)
            bodyparts_cam1_cam12_singleAni_data.to_hdf(bodyparts_cam1_cam12_singleAni,key='tracks')

            # cam2 
            bodyparts_cam2_cam12_data = pd.read_hdf(bodyparts_cam2_cam12)
            bodyparts_cam2_cam12_singleAni_data = {}
            bodyparts_cam2_cam12_singleAni_data[singlecam_ana_type]=bodyparts_cam2_cam12_data.loc[:,(singlecam_ana_type,animalname_ana)]
            bodyparts_cam2_cam12_singleAni_data=pd.concat(bodyparts_cam2_cam12_singleAni_data, axis=1)
            bodyparts_cam2_cam12_singleAni_data.to_hdf(bodyparts_cam2_cam12_singleAni,key='tracks')

            # cam3
            bodyparts_cam3_cam23_data = pd.read_hdf(bodyparts_cam3_cam23)
            bodyparts_cam3_cam23_singleAni_data = {}
            bodyparts_cam3_cam23_singleAni_data[singlecam_ana_type]=bodyparts_cam3_cam23_data.loc[:,(singlecam_ana_type,animalname_ana)]
            bodyparts_cam3_cam23_singleAni_data=pd.concat(bodyparts_cam3_cam23_singleAni_data, axis=1)
            bodyparts_cam3_cam23_singleAni_data.to_hdf(bodyparts_cam3_cam23_singleAni,key='tracks')


            ## triangulation without filtering, should take < 15 seconds

            # for single animal
            fname_dict = {
                '1': bodyparts_cam1_cam12_singleAni,
                '2': bodyparts_cam2_cam12_singleAni,
                '3': bodyparts_cam3_cam23_singleAni,
            }
           
            d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())

            score_threshold = 0.1

            n_cams, n_points, n_joints, _ = d['points'].shape
            points = d['points']
            scores = d['scores']

            bodyparts = d['bodyparts']

            # remove points that are below threshold
            points[scores < score_threshold] = np.nan
  
            points_flat = points.reshape(n_cams, -1, 2)
            scores_flat = scores.reshape(n_cams, -1)

            p3ds_flat = cgroup.triangulate(points_flat, progress=True)
            reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)

            p3ds = p3ds_flat.reshape(n_points, n_joints, 3)
            reprojerr = reprojerr_flat.reshape(n_points, n_joints)


            ## save the new h5 files after 3d triangulation
            bodyparts_3d_singleAni_anipose = {}
            
            nbodyparts = np.shape(bodyparts)[0]
            for ibodypart in np.arange(0,nbodyparts,1):
                # remove outlier
                ind_outlier_x = (p3ds[:,ibodypart,0]<(np.nanmean(p3ds[:,ibodypart,0])-2*np.nanstd(p3ds[:,ibodypart,0])))|(p3ds[:,ibodypart,0]>(np.nanmean(p3ds[:,ibodypart,0])+2*np.nanstd(p3ds[:,ibodypart,0])))
                p3ds[ind_outlier_x,ibodypart,0]=np.nan
                ind_outlier_y = (p3ds[:,ibodypart,1]<(np.nanmean(p3ds[:,ibodypart,1])-2*np.nanstd(p3ds[:,ibodypart,1])))|(p3ds[:,ibodypart,1]>(np.nanmean(p3ds[:,ibodypart,1])+2*np.nanstd(p3ds[:,ibodypart,1])))
                p3ds[ind_outlier_y,ibodypart,1]=np.nan
                ind_outlier_z = (p3ds[:,ibodypart,2]<(np.nanmean(p3ds[:,ibodypart,2])-2*np.nanstd(p3ds[:,ibodypart,2])))|(p3ds[:,ibodypart,2]>(np.nanmean(p3ds[:,ibodypart,2])+2*np.nanstd(p3ds[:,ibodypart,2])))
                p3ds[ind_outlier_z,ibodypart,2]=np.nan

                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'x')] = p3ds[:,ibodypart, 0]
                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'y')] = p3ds[:,ibodypart, 1]
                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'z')] = p3ds[:,ibodypart, 2]
            
            bodyparts_3d_singleAni_anipose_merge[ianimal_ana] = pd.DataFrame(bodyparts_3d_singleAni_anipose)


        ## combine the two animals 
 
        bodyparts_3d_anipose = pd.concat([bodyparts_3d_singleAni_anipose_merge[0],bodyparts_3d_singleAni_anipose_merge[1]],axis=1)

        # save the combine the two animal 3d file
        bodyparts_3d_anipose.to_hdf(bodyparts_3d_anipose_file,key='tracks')  

    else:
        ## load saved 3d_anipose h5 file
        bodyparts_3d_anipose = pd.read_hdf(bodyparts_3d_anipose_file)




    ## Section 3 - plot the demo videos
    if do_videodemos:

        print('make the demo Anipose video for '+animalname_ana)

        import sys
        sys.path.append('/home/ws523/marmoset_tracking_DLCv2/following_up_analysis/3d_recontruction_analysis_self_and_coop_task/ana_functions')
        from tracking_video_3d_demo import tracking_video_3d_demo

        session_start_time = session_start_times[idate]
        fps = 30 
        nframes = 60*fps

        animal1_filename = animal1
        animal2_filename = animal2

        add_date_dir = twocamera_videos_cam12+'/anipose_cam123_3d_demo_videos/'+date_tgt+'_'+animal1_filename+'_'+animal2_filename
        if not os.path.exists(add_date_dir):
            os.makedirs(add_date_dir)
        video_file = add_date_dir+'/'+date_tgt+'_'+animal1_filename+'_'+animal2_filename+'_anipose_3d_tracking_demo.mp4'
        tracking_video_3d_demo(bodyparts_3d_anipose['weikang'],animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename, animal2_filename,session_start_time,fps,nframes,video_file)

      
