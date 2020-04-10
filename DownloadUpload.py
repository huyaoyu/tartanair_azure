
# Author: Yaoyu Hu <yyhu_live@outlook.com>

import argparse
import multiprocessing
from queue import Empty
import shutil
import time
import os
import zipfile

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

from left2all import left2all

STAT_OK   = 0
STAT_FAIL = 1

MSG_DELIMITER = "|"

def cprint(msg, flagSilent=False):
    if ( not flagSilent ):
        print(msg)

def get_container_client(connectionStr, containerName):
    serviceClient = BlobServiceClient.from_connection_string(connectionStr)
    containerClient = serviceClient.get_container_client(containerName)

    return containerClient, serviceClient

def get_azure_container_client():
    # Get the connection string from the environment variable.
    connectStr = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    print(connectStr)

    print("Get the container client. ")
    cClient, _ = get_container_client( connectStr, "tartanairdataset" )

    return cClient

def get_local_file_list(prefix, suffix, n=1000):
    return [ "%s%06d%s" % (prefix, i, suffix) for i in range(n) ]

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
    
    # Test directory.
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

def generate_name_lists(fn, testNum=0):
    infile0 = read_name_list(fn)

    infile1 = []

    for f in infile0:
        parts = get_filename_parts(f)
        infile1.append( "%s/%s_target%s" % ( parts[0], parts[1], parts[2] ) )
    
    nameList0 = []
    for f in infile0:
        names = read_name_list(f)
        nameList0 = nameList0 + names

    nameList1 = []
    for f in infile1:
        names = read_name_list(f)
        nameList1 = nameList1 + names

    N = len(nameList0)

    assert( N == len(nameList1) )

    if ( testNum > 0 ):
        if ( testNum < N ):
            N = testNum

    nameList0 = nameList0[:N]
    nameList1 = nameList1[:N]

    return nameList0, nameList1

def prepare_filenames(nameList0, nameList1, idx, skip, outDir, testNum, flagFlow=False):
    N0 = len(nameList0)
    N1 = len(nameList1)

    if ( flagFlow ):
        N0 = N0 - 1
        N1 = N1 - 1

    assert(N0 > 0)
    assert(N1 > 0)
    assert(idx >= 0)

    # Generate filenames.
    files0 = []
    files1 = []

    if ( testNum > 0 ):
        if ( testNum < N0 ):
            N0 = testNum
            N1 = testNum

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

def process_single_file(name, cClient, jobStrList, flagSilent=False):
    """
    name is the name of the process.
    """

    startTime = time.time()

    ret = STAT_OK

    s = "%s: from %s to %s. %s " % ( name, jobStrList[0], jobStrList[1], MSG_DELIMITER )

    cprint("%s. " % (jobStrList[0]), flagSilent)

    # Download the blob file.
    try:
        bClient = cClient.get_blob_client(blob=jobStrList[0])
        data = bClient.download_blob()

        outFn = jobStrList[1]

        # Write the data to the file system.
        fp = open( outFn, "wb" )
        fp.write(data.content_as_bytes())
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
        fp.close()

    endTime = time.time()
    
    s = s + "%ds for processing. %s " % (endTime - startTime, MSG_DELIMITER )

    cprint(s, flagSilent)
    cprint("%s: " % (name), flagSilent)

    return [ret, s]

def worker(name, q, p, rq, cClient, flagSilent=False):
    """
    name: String, the name of this worker process.
    q: A JoinableQueue.
    p: A pipe connection object. Only for receiving.
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

            ret, s = process_single_file(name, cClient, jobStrList, flagSilent)

            rq.put([ret, s], block=True)

            q.task_done()
        except Empty as exp:
            pass
    
    cprint("%s: Work done." % (name), flagSilent)

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

    parser.add_argument("--skip", type=int, default=0, \
        help="The number of characters to skip for the input file name. Non-negative.")

    parser.add_argument("--flow", action="store_true", default=False, \
        help="Set this flag for optical flow. ")

    parser.add_argument("--np", type=int, default=2, \
        help="The number of processes.")

    parser.add_argument("--test-n", type=int, default=0, \
        help="The number of poses to process for testing. Set 0 to disable.")

    args = parser.parse_args()

    assert args.np > 0

    return args

if __name__ == "__main__":
    startTime = time.time()
    
    # Parse the arguments.
    args = parse_args()

    print("Main: Main process.")

    # Generate name lists.
    nameList0, nameList1 = generate_name_lists(args.infile0, args.test_n)

    # # Read the name list file.
    # nameList0 = read_name_list(args.infile0)
    # nameList1 = read_name_list(args.infile1)

    # Prepare the filenames.
    files0, files1 = prepare_filenames(
        nameList0, nameList1, 
        args.idx, args.skip, args.outdir, args.test_n, args.flow)

    nFiles = len(files0)

    # Get the Azure container client.
    cClient = get_azure_container_client()

    # Prepare for the job queue.
    jobQ    = multiprocessing.JoinableQueue()
    resultQ = multiprocessing.Queue()

    processes = []
    pipes     = []

    print("Main: Create %d processes." % (args.np))

    for i in range(int(args.np)):
        [conn1, conn2] = multiprocessing.Pipe(False)
        processes.append( multiprocessing.Process( \
            target=worker, args=["P%03d" % (i), jobQ, conn1, resultQ, cClient, True]) )
        pipes.append(conn2)

    for p in processes:
        p.start()

    print("Main: All processes started.")

    for i in range(nFiles):
        jobQ.put([ files0[i], "%s/%s" % ( args.outdir, files1[i] ) ])

    print("Main: All jobs submitted.")

    resultList = []
    resultCount = 0

    while(resultCount < nFiles):
        try:
            print("Main: Get index %d. " % (resultCount))
            r = resultQ.get(block=True, timeout=1)
            resultList.append(r)
            resultCount += 1
        except Empty as exp:
            if ( resultCount == nFiles ):
                print("Main: Last element of the result queue is reached.")
                break

    jobQ.join()

    print("Main: Queue joined.")

    for p in pipes:
        p.send("exit")

    print("Main: Exit command sent to all processes.")

    for p in processes:
        p.join()

    print("Main: All processes joined.")

    print("Main: Starts process the result.")

    try:
        resFn = "%s/%s_Log.txt" % ( args.outdir, args.zipname )
        fp = open(resFn, "w")
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

    # Zip.
    print("Main: Start zipping...")
    zipFn = "%s/%s.zip" % (args.outdir, args.zipname)
    z = zipfile.ZipFile(zipFn, "w")
    for f in files1:
        z.write( "%s/%s" % (args.outdir, f) )
    
    z.close()

    # Delete the temporary files.
    prefix = get_leading_directory(files1[0])
    removeDir = "%s/%s" % ( args.outdir, prefix )
    removeDir = removeDir.strip()

    # Check if removeDir is /.
    if ( "/" == removeDir ):
        raise Exception("Remove /. ")

    print("Main: Removing the temporary files... ")
    shutil.rmtree(removeDir)

    endTime = time.time()

    print("Main: Job done. Total time is %ds." % (endTime - startTime))
