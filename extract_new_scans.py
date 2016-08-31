#!/usr/bin/env python
"""
Searches the given directory for zipped scans and extracts the scans to
a folder that matches the PatientName field of the dicom headers
(if such a directory exists).

If a given scan has already been extracted that scan will be skipped, so
this program is safe to run on a <scans_dir> that continuously updates with new
scans.

Usage:
    extract_new_scans.py [options] <scans_dir> <extract_dir>

Arguments:
    <scans_dir>         Path to the folder containing all zipped
                        scans to extract (will skip scans that have already
                        been extracted to <extract_dir>)

    <extract_dir>       Location of parent directory for all the extracted scans.
                        Each scan to be extracted is expected to have a folder in
                        <extract_dir> matching the PatientName field of its
                        dicom headers (unless the --make-folders option is set).
Options:
    --make-folders      Create folders, if needed, that match the desired
                        path for the extracted scans.

    --add-path STR      Appends STR to the path once a folder is found. The
                        resulting path for extraction will be of the form
                        <extract_dir>/<PatientName>/<STR>

    -v, --verbose       Print messages to the terminal about which scans are
                        being extracted and to where
"""

from docopt import docopt
import os
import sys
import glob
import dicom as dcm
import zipfile
import tempfile
import shutil
import contextlib

VERBOSE = False

def error_message(msg, continue_exec=True):
    print("ERROR: " + msg)
    sys.stdout.flush()
    if not continue_exec:
        sys.exit(1)

def verbose_message(msg):
    if VERBOSE:
        print(msg)

def main():
    global VERBOSE
    arguments   = docopt(__doc__)
    scans_dir   = arguments['<scans_dir>']
    extract_dir = arguments['<extract_dir>']
    make_dirs   = arguments['--make-folders']
    path_ext    = arguments['--add-path']
    VERBOSE     = arguments['--verbose']

    if path_ext is None:
        path_ext = ""

    scans_dir = sanitize_path(scans_dir)
    extract_dir = sanitize_path(extract_dir)

    for scan in glob.glob("{}/*.zip".format(scans_dir)):
        scan_id = get_scan_id(scan)

        if scan_id is None:
            # Not a folder containing dicoms, skip it.
            continue

        output_path = os.path.join(extract_dir, scan_id, path_ext)

        if not already_extracted(scan, output_path):
            verbose_message("Found new scan: {}".format(scan))
            extract_scan(scan, output_path, make_dirs)

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

def get_scan_id(scan):
    """
    Checks the contents of the zipped scan and returns the value of the
    PatientName field for the first dicom found. Returns None if no dicoms
    are found.
    """
    scan_id = None
    try:
        with read_zip(scan) as zip_scan:
            for item in zip_scan.namelist():
                if ".dcm" in item:
                    scan_id = read_patient_name(zip_scan, item)
                    break
    except:
        verbose_message("{} is not a readable zipfile".format(os.path.basename(scan)))

    return scan_id

@contextlib.contextmanager
def read_zip(zip_file):
    open_zip = zipfile.ZipFile(zip_file, 'r')
    try:
        yield open_zip
    finally:
        open_zip.close()

def read_patient_name(zip_scan, dicom):
    """
    Returns the PatientName field of the given dicom
    """
    with make_temp_dir() as tmp:
        zip_scan.extract(dicom, tmp)
        image = os.path.join(tmp, dicom)
        dicom = dcm.read_file(image)
        name = dicom.PatientName
    return name

@contextlib.contextmanager
def make_temp_dir():
    temp = tempfile.mkdtemp()
    try:
        yield temp
    finally:
        shutil.rmtree(temp)

def already_extracted(scan, target_dir):
    fname = os.path.basename(scan)
    unzip_name = os.path.splitext(fname)[0]
    extract_loc = os.path.join(target_dir, unzip_name)
    if os.path.isdir(extract_loc):
        return True
    return False

def extract_scan(scan, output_path, make_dirs):
    if make_dirs:
        try:
            os.makedirs(output_path)
        except:
            error_message("Cannot make {}".format(output_path))

    if os.path.isdir(output_path):
        verbose_message("extracting {} to {}".format(os.path.basename(scan),
                            output_path))
        with read_zip(scan) as zip_scan:
            zip_scan.extractall(output_path)
    else:
        error_message("{} doesn't exist. Cannot extract {}".format(output_path,
                os.path.basename(scan)))

if __name__ == '__main__':
    main()
