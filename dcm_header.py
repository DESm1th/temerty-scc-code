#!/usr/bin/env python
"""
Reads a dicom's header and prints it or a specific field from it

Usage:
    dcm_headers.py [options] <dcm_path>

Arguments:
    <dcm_path>          The path to the dicom to be examined.

Options:
    --field STR         A string matching the name of a specific header
                        field (i.e. PatientName or SeriesDescription)
"""

import os
import sys
from docopt import docopt
import dicom as dcm

def main():
    arguments   = docopt(__doc__)
    dcm_path    = arguments['<dcm_path>']
    field       = arguments['--field']

    dcm_path = sanitize_path(dcm_path)
    dcm_header = read_dcm(dcm_path)

    if field is not None:
        display_dcm_field(dcm_header, field)
    else:
        print(dcm_header)

def sanitize_path(user_path):
    """
    Ensures an absolute and normalized path is always used so path dependent
    functions don't mysteriously fail

    os.path.abspath is not used, because symbolic links may cause a broken
    path to be generated.
    """
    curr_path = os.environ['PWD']
    abs_path = os.path.join(curr_path, user_path)
    clean_path = os.path.normpath(abs_path)

    return clean_path

def read_dcm(path):
    try:
        header = dcm.read_file(path)
    except:
        print("{} cannot be read".format(path))
        sys.exit(1)

    return header

def display_dcm_field(header, field):
    try:
        print(getattr(header, field))
    except AttributeError:
        print("{} is not a header field. Check spelling.".format(field))

if __name__ == '__main__':
    main()
