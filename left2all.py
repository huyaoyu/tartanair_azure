def left2all(leftfilepath):

    folders = ['depth_left', 
    'depth_right', 
    'image_left', 
    'image_right', 
    'seg_left', 
    'seg_right'] 

    surfixes = ["_left_depth.npy",
    "_right_depth.npy",
    "_left.png",
    "_right.png",
    "_left_seg.npy",
    "_right_seg.npy"] 

    # handle flow files
    flowfolder = 'flow'
    flow_surfix = "_flow.npy"
    flow_mask_surfix = "_mask.npy"

    filelist = []

    for folder, surfix in zip(folders, surfixes): 
        newfile = leftfilepath.replace('image_left', folder) + surfix
        filelist.append(newfile)

    indstr = leftfilepath.split('/')[-1]
    flow_str = indstr + '_' + str(int(indstr)+1).zfill(6)
    flow_prefix = leftfilepath.split('image_left')[0] + flowfolder + '/' + flow_str
    flowfile = flow_prefix + flow_surfix
    maskfile = flow_prefix + flow_mask_surfix
    filelist.append(flowfile)
    filelist.append(maskfile)

    return filelist

if __name__ == '__main__':
    print( left2all('/data/datasets/wenshanw/tartan_data/abandonedfactory_night/Data/P000/image_left/000000') )
