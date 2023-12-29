# spike analysis function

def spike_analysis_FR_calculation(fs_spikes, spike_clusters_data, spike_time_data):
    
    clusters_unique = np.unique(spike_clusters_data)
    nclusters = np.shape(clusters_unique)[0]

    spike_time_allclusters = dict.fromkeys(np.char.mod('%d', clusters_unique),[])
    FR_allclusters = dict.fromkeys(np.char.mod('%d', clusters_unique),[])

    for icluster in np.arange(0,nclusters,1):

        ind = spike_clusters_data == clusters_unique[icluster]

        # spike time stamps
        spike_time_icluster = spike_time_data[ind]
        spike_time_allclusters[str(clusters_unique[icluster])] = spike_time_icluster/fs_spikes # change the unit to second

        # firing rate
        xxx = spike_time_icluster
        # xxx_plot = np.linspace(0, total_session_time*fs_spikes, int(total_session_time/FR_kernel))
        xxx_plot = np.linspace(0, total_session_time*fs_spikes, int(total_session_time*fs_spikes/200))
        kde = KernelDensity(kernel="gaussian", bandwidth=FR_kernel*fs_spikes).fit(xxx.reshape(-1, 1)) # 100ms bandwith gaussian kernel
        log_dens = kde.score_samples(xxx_plot.reshape(-1, 1))
        FR_icluster = np.exp(log_dens)*fs_spikes
        
        #
        FR_allclusters[str(clusters_unique[icluster])] = FR_icluster
        
    # FR_timepoint_allclusters = xxx_plot/fs_spikes
    FR_timepoint_allclusters = xxx_plot
    
    return spike_time_allclusters,FR_timepoint_allclusters,FR_allclusters
        
