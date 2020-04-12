
import argparse
import glob
import numpy as np
import os

import FilesystemUtils as fu

def print_list(li):
    for item in li:
        print(item)

def write_list_2_file(fn, li):
    np.savetxt(fn, li, fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory list files. ")

    parser.add_argument("indir", type=str, 
        help="The input directory. ")

    parser.add_argument("outdir", type=str, 
        help="The output directory. ")

    parser.add_argument("pattern", type=str, 
        help="The search pattern. ")

    parser.add_argument("--out-prefix", type=str, default="", 
        help="The filename prefix for the output files. ")

    parser.add_argument("--out-name", type=str, default="TrajectoryList.txt", 
        help="The output file name. ")

    args = parser.parse_args()

    # Test the output directory.
    fu.test_directory(args.outdir)

    # Find all the root directories in args.indir.
    roots = [ p \
        for p in os.listdir(args.indir) \
            if os.path.isdir( os.path.join(args.indir, p) ) ]

    roots = sorted( roots )

    print("Root directories are: ")
    print_list(roots)

    print("Process all root directories. ")
    for p in roots:
        # Get the full path of the sub directory.
        root = os.path.join(args.indir, p)

        sp = "%s/**/%s" % (root, args.pattern)

        files = sorted( glob.glob( sp, recursive=True ) )

        if ( 0 == len(files) ):
            print("No files found with sp = %s. " % (sp))
            continue
        else:
            print("%3d files found for root %s. " % (len(files), p))

        # print_list(files)

        # Compose the output file name.
        outFn = "%s/%s/%s%s" % ( args.outdir, p, args.out_prefix, args.out_name )

        # Test the output directory.
        fu.test_directory_by_filename(outFn)

        # Save the list to the output file.
        write_list_2_file(outFn, files)

    print("Done.")
    