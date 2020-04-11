
# Author: Yaoyu Hu <yyhu_live@outlook.com>

import argparse
import multiprocessing
import numpy as np
from queue import Empty
import shutil
import time
import os
import zipfile

# Azure packages.
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Local modules and pacakges.
from left2all import left2all
from PoseInfo import get_posefile_names

# Global flags.
STAT_OK   = 0
STAT_FAIL = 1

# Global message delimiter.
MSG_DELIMITER = "|"

def ceil_integer(val, base):
    ceilVal = int( np.ceil( 1.0*val/base ) * base )
    return ceilVal, ceilVal - val 

def cprint(msg, flagSilent=False):
    if ( not flagSilent ):
        print(msg)

def get_container_client(connectionStr, containerName):
    serviceClient = BlobServiceClient.from_connection_string(connectionStr)
    containerClient = serviceClient.get_container_client(containerName)

    return containerClient, serviceClient

def get_azure_container_client(envString, containerString):
    # Get the connection string from the environment variable.
    connectStr = os.getenv(envString)
    # print(connectStr)

    # print("Get the container client. ")
    cClient, _ = get_container_client( connectStr, containerString )

    return cClient

def get_filename_parts(fn):
    p = os.path.split(fn)

    if ( "" == p[0] ):
        p = (".", p[1])

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]

def get_leading_directory(fn):
    n = fn.find("/")

    if ( -1 == n ):
        return "."
    else:
        return fn[:n]

def test_directory(d):
    if ( not os.path.isdir(d) ):
        os.makedirs(d)

def test_directory_by_filename(fn):
    # Get the directory.
    parts = get_filename_parts(fn)
    
    # Test the directory.
    test_directory(parts[0])

def read_name_list(fn):
    if ( not os.path.isfile(fn) ):
        raise Exception("%s does not exist. " % (fn))
    
    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

    if ( 0 == len(lines) ):
        raise Exception("%s contains no content. " % (fn))

    for i in range( len(lines) ):
        lines[i] = lines[i].strip()

    return lines

def generate_name_lists(fn, flagFlow=False, testNum=0 ):
    infile0 = read_name_list(fn)

    infile1 = []

    for f in infile0:
        parts = get_filename_parts(f)
        infile1.append( "%s/%s_target%s" % ( parts[0], parts[1], parts[2] ) )
    
    nameList0 = []
    nameList1 = []
    poseParams = []

    for i in range( len(infile0) ):
        names0 = read_name_list(infile0[i])
        if ( flagFlow ):
            nameList0 = nameList0 + names0[:-1]
        else:
            nameList0 = nameList0 + names0

        names1 = read_name_list(infile1[i])
        if ( flagFlow ):
            nameList1 = nameList1 + names1[:-1]
        else:
            nameList1 = nameList1 + names1

        poseParams.append( {
            "sourceStart": names0[0],
            "sourceEnd": names0[-1],
            "targetStart": names1[0] } )

    N = len(nameList0)

    assert( N == len(nameList1) )

    poseFileExpNum = N # The expected number of entries for poses.
    if ( flagFlow ):
        poseFileExpNum = poseFileExpNum + len(infile0)

    if ( testNum > 0 ):
        if ( testNum < N ):
            N = testNum

    nameList0 = nameList0[:N]
    nameList1 = nameList1[:N]

    return poseFileExpNum, nameList0, nameList1, poseParams

def prepare_filenames(nameList0, nameList1, idx, skip, outDir):
    N0 = len(nameList0)
    N1 = len(nameList1)

    assert(N0 > 0)
    assert(N1 > 0)
    assert(idx >= 0)

    # Generate filenames.
    files0 = []
    files1 = []

    for i in range(N0):
        names0 = left2all(nameList0[i])
        name0  = names0[idx][skip:]
        files0.append( name0 )

        names1 = left2all(nameList1[i])
        name1  = names1[idx][skip:]
        files1.append( name1 )

        tempFn = "%s/%s" % ( outDir, name1 )
        test_directory_by_filename( tempFn )

    return files0, files1

def main_write_result_queue_2_file(fn, resultList):
    try:
        test_directory_by_filename(fn)

        fp = open(fn, "w")
        for res in resultList:
            fp.write("%d %s %s\n" % ( res[0], MSG_DELIMITER, res[1] ))
    except OSError as ex:
        print("Main: Open %s failed. " % (resFn))
    except Exception as ex:
        print("Main: Exeption: %s. " % (str(ex)))
        fp.close()
    else:
        print("Main: Results written. ")
        fp.close()

def create_numpy_object_from_blob(cClient, blob, dtype):
    tempFn = "/tmp/TempNumpyTxt.txt"
    
    try:
        bc = cClient.get_blob_client(blob=blob)
        data = bc.download_blob()

        # Write the the filesystem temporarily.
        fp = open( tempFn, "wb" )
        byteData = data.content_as_bytes()
        fp.write(byteData)
        fp.close()

    except ResourceNotFoundError as ex:
        raise Exception("%s not found on the server. " % (blob))
    except OSError as ex:
        raise Exception( "Cannot open %s. " % ("/tmp/TempNumpyTxt.txt") )
    except Exception as ex:
        fp.close()
        raise Exception("Exeption: %s. File closed. " % ( str(ex) ))

    # Open the NumPy object again.
    array = np.loadtxt(tempFn, dtype=dtype)

    # Remove the temporary file.
    os.remove(tempFn)

    return array

def get_pose_objects(cClient, poseParams):
    """
    poseParams: A list of dicts. A dict contains the following key values:
        sourceStart, sourceEnd, targetStart.
    """

    poses = []

    totalSize = 0

    for param in poseParams:
        sourceLeftFile, sourceRightFile, \
        startIdx, endIdx, targetLeftFile, targetRightFile = \
            get_posefile_names( param["sourceStart"], param["sourceEnd"], param["targetStart"] )

        startIdx = int(startIdx)
        endIdx = int(endIdx)

        # Create the numpy objects.
        poseLeft = create_numpy_object_from_blob(cClient, sourceLeftFile, np.float32)
        poseLeft = poseLeft[startIdx:endIdx+1]

        poseRight = create_numpy_object_from_blob(cClient, sourceRightFile, np.float32)
        poseRight = poseRight[startIdx:endIdx+1]

        totalSize += poseLeft.shape[0]

        poses.append({
            "poseLeft": poseLeft,
            "poseRight": poseRight,
            "targetLeftFile": targetLeftFile,
            "targetRightFile": targetRightFile })

    return poses, totalSize

def write_numpy(fn, array):
    parts = get_filename_parts(fn)

    test_directory(parts[0])

    if ( ".npy" == parts[2] ):
        np.save(fn, array)
    else:
        np.savetxt(fn, array)

def write_poses(poses, localDir):
    """
    poses: A list of dicts. A dict contains the following keys:
        poseLeft, poseRight, targetLeftFile, targetRightFile. 
    """

    for pose in poses:
        outFn = "%s/%s" % ( localDir, pose["targetLeftFile"] )
        write_numpy(outFn, pose["poseLeft"])

        outFn = "%s/%s" % ( localDir, pose["targetRightFile"] )
        write_numpy(outFn, pose["poseRight"])

def upload_file_2_blob(cc, blob, file, flagOverwrite=True):
    try:
        # Get the blob.
        bc = cc.get_blob_client(blob=blob)

        # Read the file in binary.
        fp = open( file, "rb" )

        # Upload.
        bc.upload_blob(fp, blob_type="BlockBlob", overwrite=flagOverwrite)

        # Clean up.
        fp.close()
    except ResourceNotFoundError as ex:
        # Should never be here.
        raise Exception("Exception: %s" % (str(ex)))
    except ResourceExistsError as ex:
        raise Exception("Exception: %s\nBlob %s already exits. flagOverwrite = False." % (str(ex), blob))
    except OSError as ex:
        raise Exception("Cannot open %s. " % ( file ))
    except Exception as ex:
        fp.close()
        raise Exception("Exeption: %s. File closed. " % ( str(ex) ))

def upload_pose_files(cc, poses, localDir):
    """
    poses: A list of dicts. A dict contains the following keys:
        poseLeft, poseRight, targetLeftFile, targetRightFile. 
    """

    for pose in poses:
        blob = pose["targetLeftFile"]
        fn   = "%s/%s" % ( localDir, blob )
        upload_file_2_blob(cc, blob, fn)

        blob = pose["targetRightFile"]
        fn   = "%s/%s" % ( localDir, blob )
        upload_file_2_blob(cc, blob, fn)

def process_single_file( name, 
        barrierDownload, barrierUpload,
        cClient, jobStrList, 
        flagUpload=False, ccu=None, 
        npyForceConvert=False, 
        flagSilent=False ):
    """
    name is the name of the process.
    """

    startTime = time.time()

    ret = STAT_OK

    if ( "Barrier" == jobStrList[3] ):
        cprint("%s: Enter barrier mode. " % (name), flagSilent)

    #     bIdx = barrierDownload.wait()
    #     if ( 0 == bIdx ):
    #         barrierDownload.reset()

    #     if ( flagUpload ):
    #         bIdx = barrierUpload.wait()
    #         if ( 0 == bIdx ):
    #             barrierUpload.reset()
        
        endTime = time.time()
    
        s = "%s: Barrier. %ds for processing. %s " % (name, endTime - startTime, MSG_DELIMITER )

        return [ret, s]

    s = "%s: from %s to %s. %s " % ( name, jobStrList[0], jobStrList[1], MSG_DELIMITER )

    cprint("%s. " % (jobStrList[0]), flagSilent)

    # Download the blob file.
    try:
        bClient = cClient.get_blob_client(blob=jobStrList[0])

        # bIdx = barrierDownload.wait()
        # if ( 0 == bIdx ):
        #     barrierDownload.reset()

        # print("%s: Begin download. " % ( name ))
        data = bClient.download_blob()
        # print("%s: End download. " % ( name ))

        outFn = "%s/%s" % ( jobStrList[2], jobStrList[1])

        parts = get_filename_parts( jobStrList[0] )

        # Write the data to the file system.
        fp = open( outFn, "wb" )
        byteData = data.content_as_bytes()
        fp.write(byteData)
        fp.close()

        # Special treatment for .npy file.
        if ( npyForceConvert and ".npy" == parts[2] ):
                # Open the file, convert, and save again.
                array = np.load(outFn)
                array = array.astype(np.float32)
                np.save(outFn, array)

        # Check if we have to upload the file.
        if ( flagUpload ):
            # Get the blob even it is not exist.
            bcu = ccu.get_blob_client(blob=jobStrList[1])
            
            if ( ".npy" == parts[2] ):
                # Open the file.
                fp = open( outFn, "rb" )

                # bIdx = barrierUpload.wait()
                # if ( 0 == bIdx ):
                #     barrierUpload.reset()

                # print("%s: Before upload. " % (name))
                bcu.upload_blob(fp, blob_type="BlockBlob", overwrite=True) 
                # print("%s: After upload. " % (name))
                fp.close()
            else:
                # bIdx = barrierUpload.wait()
                # if ( 0 == bIdx ):
                #     barrierUpload.reset()
                
                # print("%s: Before upload. " % (name))
                bcu.upload_blob(byteData, blob_type="BlockBlob", overwrite=True)
                # print("%s: After upload. " % (name))
    except ResourceNotFoundError as ex:
        s = s + "Azure exception: %s. %s " % ( str(ex), MSG_DELIMITER )
        ret = STAT_FAIL
    except OSError as ex:
        s = s + "Exeption: %s. %s " % ( str(ex), MSG_DELIMITER )
        ret = STAT_FAIL
    except Exception as ex:
        s = s + "Exeption: %s. File closed. %s " % ( str(ex), MSG_DELIMITER )
        ret = STAT_FAIL
        fp.close()
    else:
        pass

    endTime = time.time()
    
    s = s + "%ds for processing. %s " % (endTime - startTime, MSG_DELIMITER )

    cprint(s, flagSilent)
    cprint("%s: " % (name), flagSilent)

    return [ret, s]

def worker(name, q, p, rq,
        barrierDownload, barrierUpload,
        cClient, 
        flagUpload=False, ccu=None, 
        npyForceConvert=False, 
        flagSilent=False):
    """
    name: String, the name of this worker process.
    q: A JoinableQueue.
    p: A pipe connection object. Only for receiving.
    ccu: Container client for uploading.
    """

    cprint("%s: Worker starts." % (name), flagSilent)

    while (True):
        if (p.poll()):
            command = p.recv()

            cprint("%s: %s command received." % (name, command), flagSilent)

            if ("exit" == command):
                break

        try:
            jobStrList = q.get(True, 1)
            # print("{}: {}.".format(name, jobStrList))

            ret, s = process_single_file(name, 
                barrierDownload, barrierUpload,
                cClient, jobStrList, 
                flagUpload, ccu, npyForceConvert, flagSilent)

            rq.put([ret, s], block=True)

            q.task_done()
        except Empty as exp:
            pass
    
    cprint("%s: Work done." % (name), flagSilent)

class dummy_args(object):
    def __init__(self, infile0, outdir, zipname, idx):
        self.infile0 = infile0
        self.outdir  = outdir
        self.zipname = zipname
        self.idx     = idx

        self.zip          = False
        self.skip         = 0
        self.flow         = False
        self.remove_temporary_files = False
        self.remove_zip   = False
        self.upload       = False
        self.download_env = "AZURE_STORAGE_CONNECTION_STRING"
        self.upload_env   = "AZURE_STORAGE_CONNECTION_STRING"
        self.download_c   = "tartanairdataset"
        self.upload_c     = "tartanair-release0"
        self.upload_zip_overwrite = False
        self.npy_force_convert    = False
        self.np           = 2
        self.main_prefix  = ""
        self.test_n       = 0
        self.disable_child_silent = False

def parse_args():
    parser = argparse.ArgumentParser(description="Filter the files.")

    parser.add_argument("infile0", type=str, \
        help="The the file name of the name list file.")

    # parser.add_argument("infile1", type=str, \
    # help="The the file name of the name list file.")

    parser.add_argument("outdir", type=str, \
        help="The output directory.")

    parser.add_argument("zipname", type=str, \
        help="The filename of the zip file not include the .zip extension.")

    parser.add_argument("idx", type=int, \
        help="The index in the file types. Non-negative.")

    parser.add_argument("--zip", action="store_true", default=False, \
        help="Set this flag to enable the zipping.")

    parser.add_argument("--skip", type=int, default=0, \
        help="The number of characters to skip for the input file name. Non-negative.")

    parser.add_argument("--flow", action="store_true", default=False, \
        help="Set this flag for optical flow. ")

    parser.add_argument("--remove-temporary-files", action="store_true", default=False, \
        help="Set this flaf to remove the temporary downloade files.")

    parser.add_argument("--remove-zip", action="store_true", default=False, \
        help="Set this flaf to remove the zipped file.")

    parser.add_argument("--upload", action="store_true", default=False, \
        help="Set this flag to enable uploading at the same time.")

    parser.add_argument("--download-env", type=str, default="AZURE_STORAGE_CONNECTION_STRING", \
        help="The environment variable for the download container.")

    parser.add_argument("--upload-env", type=str, default="AZURE_STORAGE_CONNECTION_STRING", \
        help="The environment variable for the upload container.")

    parser.add_argument("--download-c", type=str, default="tartanairdataset", \
        help="The download container name.")

    parser.add_argument("--upload-c", type=str, default="tartanair-release0", \
        help="The download container name.")

    parser.add_argument("--upload-zip-overwrite", action="store_true", default=False, \
        help="Set this flag to enable overwriting when uploading the zip file.")

    parser.add_argument("--npy-force-convert", action="strore_true", default=False, \
        help="Set this flag to force .npy files to be converted into float32.")

    parser.add_argument("--np", type=int, default=2, \
        help="The number of processes.")

    parser.add_argument("--main-prefix", type=str, default="", \
        help="The prefix of main process print.")

    parser.add_argument("--test-n", type=int, default=0, \
        help="The number of poses to process for testing. Set 0 to disable.")

    parser.add_argument("--disable-child-silent", action="store_ture", default=False, \
        help="Set this flag to disable silenting child process.")

    args = parser.parse_args()

    assert args.np > 0

    return args

def run(args):
    startTime = time.time()

    print("Main: Main process.")
    
    # Generate name lists.
    poseFileExpNum, nameList0, nameList1, poseParams = generate_name_lists(
        args.infile0, args.flow, args.test_n )

    # Prepare the filenames.
    files0, files1 = prepare_filenames(
        nameList0, nameList1, 
        args.idx, args.skip, args.outdir)

    nFiles = len(files0)

    # Get the Azure container client.
    cClient = get_azure_container_client( 
        args.download_env, args.download_c )

    # Check if we will perform uploading at the same time.
    cClientUpload = None
    if ( args.upload ):
        cClientUpload = get_azure_container_client( 
            args.upload_env, args.upload_c )

    # Prepare for the job queue.
    jobQ    = multiprocessing.JoinableQueue()
    resultQ = multiprocessing.Queue()

    # Barriers.
    barrierDownload = multiprocessing.Barrier(args.np)
    barrierUpload   = multiprocessing.Barrier(args.np)

    nJobs, bJobs = ceil_integer( nFiles, args.np ) # bJobs is the number of padding jobs for the barriers.
    print("Main: nJobs: %d, bJobs: %d. " % ( nJobs, bJobs ))

    # Create all the worker processes.
    processes = []
    pipes     = []
    print("Main: Create %d processes." % (args.np))
    for i in range(int(args.np)):
        [conn1, conn2] = multiprocessing.Pipe(False)
        processes.append( 
            multiprocessing.Process( 
                target=worker, 
                args=["P%03d" % (i), jobQ, conn1, resultQ, 
                    barrierDownload, barrierUpload, 
                    cClient, args.upload, cClientUpload, 
                    args.npy_force_convert, not args.disable_child_silent] ) )
        pipes.append(conn2)

    for p in processes:
        p.start()

    print("Main: All processes started.")

    # Submit all actual jobs.
    for i in range(nFiles):
        jobQ.put([ files0[i], files1[i], args.outdir, "null" ])

    # Submit the padding jobs.
    for i in range(bJobs):
        jobQ.put( [ "null", "null", "null", "Barrier" ] )

    # Note that nJobs = nFiles + bJobs.

    print("Main: All jobs submitted.")

    # Main process starts to handle the messages in the result queue.
    resultList = []
    resultCount = 0
    while(resultCount < nJobs):
        try:
            r = resultQ.get(block=True, timeout=1)
            resultList.append(r)
            resultCount += 1
        except Empty as exp:
            if ( resultCount == nJobs ):
                print("Main: Last element of the result queue is reached.")
                break
            else:
                print("%sMain: Wait on rq-index %d. " % (args.main_prefix, resultCount))
                time.sleep(0.5)

    # Break all barriers.
    barrierUpload.abort()
    barrierDownload.abort()

    # Main process wait untill all worker. This should be always joined with out long blocking.
    jobQ.join()

    print("Main: Queue joined.")

    # Send commands to terminate all worker processes.
    for p in pipes:
        p.send("exit")

    print("Main: Exit command sent to all processes.")

    for p in processes:
        p.join()

    print("Main: All processes joined.")

    print("Main: Starts process the result.")
    resFn = "%s/RQ_%s.txt" % ( args.outdir, args.zipname )
    main_write_result_queue_2_file( resFn, resultList )

    # Get the the pose files from the server.
    print("Main: Get pose files...........")
    poses, totalPosesSize = get_pose_objects(cClient, poseParams)

    # Write the pose files to local filesystem.
    write_poses(poses, args.outdir)

    if ( poseFileExpNum != totalPosesSize ):
        raise Exception("poseFileExpNumiles != totalPosesSize. poseFileExpNum = %d, totalPosesSize = %d. " % 
            ( poseFileExpNum, totalPosesSize ) )

    # Upload the pose files.
    if ( args.upload ):
        print("Main: Uploading the pose files.")
        upload_pose_files(cClientUpload, poses, args.outdir)

    if ( args.zip ):
        # Zip.
        print("Main: Start zipping...")
        zipFn = "%s/%s.zip" % (args.outdir, args.zipname)
        test_directory_by_filename(zipFn)
        z = zipfile.ZipFile(zipFn, "w")
        for f in files1:
            z.write( "%s/%s" % (args.outdir, f) )
        
        # Pose files.
        for pose in poses:
            z.write( "%s/%s" % ( args.outdir, pose["targetLeftFile"] ) )
            z.write( "%s/%s" % ( args.outdir, pose["targetRightFile"] ) )

        z.close()

        # Uploading the zip file.
        if ( args.upload ):
            bZipFn = "%s.zip" % (args.zipname) # Blob name.
            print("Main: Uploading %s. " % (bZipFn))
            upload_file_2_blob( cClientUpload, bZipFn, zipFn, args.upload_zip_overwrite )

        if ( args.remove_zip ):
            print("Main: Removing the zip file... ")
            if ( os.path.isfile(zipFn) ):
                os.remove(zipFn)
            else:
                print("Main: %s dose not exist. " % (zipFn))

    if ( args.remove_temporary_files ):
        print("Main: Removing the temporary files... ")

        # Remove the pose files.
        for pose in poses:
            os.remove( "%s/%s" % ( args.outdir, pose["targetLeftFile"] ) )
            os.remove( "%s/%s" % ( args.outdir, pose["targetRightFile"] ) )

        # Delete the temporary files.
        prefix = get_leading_directory(files1[0])
        removeDir = "%s/%s" % ( args.outdir, prefix )
        removeDir = removeDir.strip()

        # Check if removeDir is /.
        if ( "/" == removeDir ):
            raise Exception("Remove /. ")

        shutil.rmtree(removeDir)

    endTime = time.time()

    print("Main: Download-upload job done. Total time is %ds." % (endTime - startTime))

def main(args):
    res = STAT_OK

    try:
        run(args)
    except Exception as ex:
        res = STAT_FAIL
        print("Main: Exception: %s. Abort. " % ( str(ex) ))

    return res

if __name__ == "__main__":
    # Parse the arguments.
    args = parse_args()
    main(args)
    