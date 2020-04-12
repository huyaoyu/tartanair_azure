
import os

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