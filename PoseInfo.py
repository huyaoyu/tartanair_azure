def get_posefile_names(source_start, source_end, target_start):
    '''
    source_start: 'abandonedfactory/Data/P000/image_left/000000'
    source_end: 'abandonedfactory/Data/P000/image_left/002175'
    target_end: 'abandonedfactory/Easy/P000/image_left/002175'
    '''
    sourcedir = source_start.split('image_left')[0]
    source_left_file = sourcedir + 'pose_left.txt'
    source_right_file = sourcedir + 'pose_right.txt'
    startind = int(source_start.split('/')[-1])
    endind = int(source_end.split('/')[-1])
    targetdir = target_start.split('image_left')[0]
    target_left_file = targetdir + 'pose_left.txt'
    target_right_file = targetdir + 'pose_right.txt'
    return source_left_file, source_right_file, \
        startind, endind, \
        target_left_file, target_right_file