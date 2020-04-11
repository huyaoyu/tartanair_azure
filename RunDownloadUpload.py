
import argparse
import glob
import os

from DownloadUpload import dummy_args, STAT_OK, STAT_FAIL
from DownloadUpload import main as app

INDEX_DICT_LIST=[
    {"name": "depth_left",  "idx": 0, "flow": False, "npyFC": False},
    {"name": "depth_right", "idx": 1, "flow": False, "npyFC": False},
    {"name": "image_left",  "idx": 2, "flow": False, "npyFC": False},
    {"name": "image_right", "idx": 3, "flow": False, "npyFC": False},
    {"name": "seg_left",    "idx": 4, "flow": False, "npyFC": False},
    {"name": "seg_right",   "idx": 5, "flow": False, "npyFC": False},
    {"name": "flow_flow",   "idx": 6, "flow": True,  "npyFC": True},
    {"name": "flow_mask",   "idx": 7, "flow": True,  "npyFC": False}
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DownloadUpload.py.")

    parser.add_argument("infile", type=str, 
        help="The input file.")

    parser.add_argument("outdir", type=str, 
        help="The output directory.")

    args = parser.parse_args()

    # Get a dummy argument object.
    dArgs = dummy_args(args.infile, args.outdir, "dummy", -1)

    # Configure the dummy argument.
    dArgs.download_c = "tartanairdataset"
    dArgs.upload     = True
    dArgs.upload_c   = "tartanair-release0"
    dArgs.upload_zip_overwrite = True
    dArgs.zip        = True
    dArgs.remove_zip = True
    dArgs.remove_temporary_files = True
    dArgs.npy_force_convert      = False
    dArgs.np         = 2
    dArgs.test_n     = 5

    # Prepare statistics.
    retList = [ STAT_FAIL for i in range( len(INDEX_DICT_LIST) ) ]
    count   = 0
    flagOK  = True

    for d in INDEX_DICT_LIST:
        dArgs.zipname = d["name"]
        dArgs.idx = d["idx"]
        dArgs.flow = d["flow"]
        dArgs.npy_force_convert = d["npyFC"]

        print("\n========== Process %s. ==========\n" % (dArgs.zipname))

        ret = app(dArgs)

        if ( STAT_OK == ret ):
            pass
        elif ( STAT_FAIL == ret ):
            flagOK = False
            print("Failed with zipname = {}, idx = {}, count = {}. ".format( d["name"], d["idx"], count ))
            print("Abort current processing for input file %s. " % ( args.infile ))
            break
        else:
            print("Unexpected return value {}. ".format(ret))

        retList[count] = ret
        count += 1

    if ( flagOK ):
        print("Process OK. ")
    else:
        print("Process failed. ")
    
    print("Done.")