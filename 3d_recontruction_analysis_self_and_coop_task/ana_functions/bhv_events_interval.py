# function - interval between all behavioral events

def bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    total_time = int((totalsess_time - session_start_time))
    time_point_pull1_round = time_point_pull1.reset_index(drop = True)
    time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
    time_point_pull2_round  = time_point_pull2.reset_index(drop = True)
    time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
    time_point_onewaygaze1_round = pd.Series(oneway_gaze1).reset_index(drop = True)
    time_point_onewaygaze2_round = pd.Series(oneway_gaze2).reset_index(drop = True)
    time_point_mutualgaze1_round = pd.Series(mutual_gaze1).reset_index(drop = True)
    time_point_mutualgaze2_round = pd.Series(mutual_gaze2).reset_index(drop = True)
    time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
    time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
    time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
    time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]
    #     
    time_point_bhv_events = time_point_pull1_round
    time_point_bhv_events = time_point_bhv_events.append(time_point_pull2_round)
    time_point_bhv_events = time_point_bhv_events.append(time_point_mutualgaze1_round)
    time_point_bhv_events = time_point_bhv_events.append(time_point_mutualgaze2_round)
    time_point_bhv_events = time_point_bhv_events.append(time_point_onewaygaze1_round)
    time_point_bhv_events = time_point_bhv_events.append(time_point_onewaygaze2_round)

    time_point_bhv_events = time_point_bhv_events.reset_index(drop=True)
    time_point_bhv_events = np.sort(time_point_bhv_events)
    nevents = np.shape(time_point_bhv_events)[0]
    bhv_events_interval = time_point_bhv_events[1:nevents]-time_point_bhv_events[0:nevents-1]

    Q1 = np.quantile(bhv_events_interval,0.25)
    Q2 = np.quantile(bhv_events_interval,0.5)
    Q3 = np.quantile(bhv_events_interval,0.75)
    low_lim = Q1 - 1.5 * (Q3-Q1)
    up_lim = Q3 + 1.5 * (Q3-Q1)
    # low_lim = Q1
    # up_lim = Q3
    
    low_lim = np.round(low_lim*10)/10
    up_lim = np.round(up_lim*10)/10
    
    if low_lim < 0.1:
        low_lim = 0.1
    if up_lim <0.2:
        up_lim = 0.2  
    # if up_lim < 1:
    #     up_lim = np.max(bhv_events_interval)/2
    if up_lim > 10:
        up_lim = 10
    
    return low_lim, up_lim, bhv_events_interval
