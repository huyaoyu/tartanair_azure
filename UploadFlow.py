
# Author: Yaoyu Hu <yyhu_live@outlook.com>

import argparse
import multiprocessing
import numpy as np
from queue import Empty
import shutil
import time
import os
import sys

# Azure packages.
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Local modules and pacakges.
from FilesystemUtils import get_filename_parts, get_leading_directory, test_directory, test_directory_by_filename

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

def main_write_result_queue_2_file(fn, resultList, name='main'):
    try:
        test_directory_by_filename(fn)

        fp = open(fn, "w")
        for res in resultList:
            fp.write("%d %s %s\n" % ( res[0], MSG_DELIMITER, res[1] ))
    except OSError as ex:
        print("%s: Open %s failed. " % (name, resFn))
    except Exception as ex:
        print("%s: Exeption: %s. " % (name, str(ex)))
        fp.close()
    else:
        print("%s: Results written. " % (name))
        fp.close()

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

def process_single_file( name, 
        jobStrList, 
        ccu,  
        flagSilent=False ):
    """
    name is the name of the process.
    """

    startTime = time.time()

    ret = STAT_OK

    s = "%s: %s -> %s. %s " % ( name, jobStrList[0], jobStrList[1], MSG_DELIMITER )

    cprint("%s. " % (jobStrList[0]), flagSilent)

    # Upload file.
    try:
        # Get the blob even it is not exist.
        bcu = ccu.get_blob_client(blob=jobStrList[1])
        
        # bIdx = barrierUpload.wait()
        # if ( 0 == bIdx ):
        #     barrierUpload.reset()
        
        fp = open( jobStrList[0], "rb" )

        # print("%s: Before upload. " % (name))
        bcu.upload_blob(fp, blob_type="BlockBlob", overwrite=True)
        # print("%s: After upload. " % (name))
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

    cprint("%s: %s" % (name, s), flagSilent)

    return [ret, s]

def worker(name, q, p, rq,
        ccu, 
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
                jobStrList, 
                ccu, 
                flagSilent)

            rq.put([ret, s], block=True)

            q.task_done()
        except Empty as exp:
            pass
    
    cprint("%s: Work done." % (name), flagSilent)

def worker_rq(name, rq, nFiles, resFn, timeoutCountLimit=100):
    resultList = []
    resultCount = 0
    timeoutCount = 0

    flagOK = True

    while(resultCount < nFiles):
        try:
            r = rq.get(block=True, timeout=1)
            resultList.append(r)
            resultCount += 1

            timeoutCount = 0

            if (resultCount % 100 == 0):
                print("%s: worker_rq collected %d results. " % (name, resultCount))
        except Empty as exp:
            print("%s: Wait on rq-index %d. " % (name, resultCount))
            time.sleep(0.5)
            timeoutCount += 1

            if ( timeoutCount == timeoutCountLimit ):
                print("%s: worker_rq reaches the timeout count limit (%d). Process abort. " % \
                    (name, timeoutCountLimit))
                flagOK = False
                break

    if (flagOK):
        print("%s: All results processed. " % (name))

    if ( resultCount > 0 ):
        if ( resultCount != nFiles ):
            print("%s: resultCount = %d, nFiles = %d. " % (name, resultCount, nFiles))

        main_write_result_queue_2_file( resFn, resultList, name )

class dummy_args(object):
    def __init__( self, 
        infile, infiledir, 
        outcontainer, outprefix,
        outdir ):
        self.infile       = infile
        self.infiledir    = infiledir
        self.outcontainer = outcontainer
        self.outprefix    = outprefix
        self.outdir       = outdir

        self.acc_str_env  = "AZURE_STORAGE_CONNECTION_STRING"
        self.np           = 2
        self.main_prefix  = ""
        self.test_n       = 0
        self.disable_child_silent = False

def parse_args():
    parser = argparse.ArgumentParser(description="up load flow files.")

    parser.add_argument("infile", type=str, \
        help="The the file name of the name list file.")

    parser.add_argument("infiledir", type=str, \
        help="The the root directory of the files listed in infile.")

    parser.add_argument("outcontainer", type=str, \
        help="The output container name.")

    parser.add_argument("outprefix", type=str, \
        help="The prefix used for saveing files in Azure data lake storage.")

    parser.add_argument("outdir", type=str, \
        help="The local output directory.")

    parser.add_argument("--acc-str-env", type=str, default="AZURE_STORAGE_CONNECTION_STRING", \
        help="The environment variable for the access string.")

    parser.add_argument("--np", type=int, default=2, \
        help="The number of processes.")

    parser.add_argument("--main-prefix", type=str, default="", \
        help="The prefix of main process print.")

    parser.add_argument("--test-n", type=int, default=0, \
        help="The number of poses to process for testing. Set 0 to disable.")

    parser.add_argument("--disable-child-silent", action="store_true", default=False, \
        help="Set this flag to disable silenting child process.")

    args = parser.parse_args()

    assert args.np > 0

    return args

def run(args):
    startTime = time.time()

    print("Main: Main process.")

    # Read the filenames.
    files = read_name_list(args.infile)
    nFiles = len(files)

    # Only process limited number of files if requested.
    if ( args.test_n > 0 ):
        if ( args.test_n < nFiles ):
            nFiles = args.test_n

    if ( nFiles == 0 ):
        raise Exception("No files read from %s. " % (args.infile))

    # Get the Azure container client.
    cClient = get_azure_container_client( 
        args.acc_str_env, args.outcontainer )

    # Prepare for the job queue.
    jobQ    = multiprocessing.JoinableQueue()
    resultQ = multiprocessing.Queue()

    print("Main: nFiles: %d. " % ( nFiles ))

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
                    cClient,  
                    not args.disable_child_silent] ) )
        pipes.append(conn2)

    for p in processes:
        p.start()

    resFn = "%s/RQ.txt" % ( args.outdir )
    pWorkerRQ = multiprocessing.Process( 
        target=worker_rq,
        args=["RQ", resultQ, nFiles, resFn] )
    
    pWorkerRQ.start()

    print("Main: All processes started.")

    # Submit all actual jobs.
    for i in range(nFiles):
        fnIn = "%s/%s" % (args.infiledir, files[i])
        fnOut = "%s/%s" % (args.outprefix, files[i])
        jobQ.put([ fnIn, fnOut, "null" ])

    print("Main: All jobs submitted.")

    # Main wait all results to be processed.
    pWorkerRQ.join()

    print("Main: pWorkerRQ joined. ")

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

    endTime = time.time()

    print("Main: Upload job done. Total time is %ds." % (endTime - startTime))

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
    sys.exit(main(args))
    