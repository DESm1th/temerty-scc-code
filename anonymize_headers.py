#!/usr/bin/env python
"""
Removes personally identifying information from the headers of scan folder of
dicoms.

By default this program looks in the PatientName field of the dicom headers
for the correct ID and ensures that this is copied over into the PatientID
field. The --PatientID and the --provideID options override this default.

Usage:
    anonymize_headers.py [options] <scan_path>

Arguments:
    <scan_path>                 Path to the scan folder of dicoms

Options:

    --patientID                 Use the PatientID field of the dicoms as the
                                'correct' ID and replace the PatientName. Will be
                                ignored if --provideID is also set

    --provideID STR             Supply the ID to place in both the PatientID
                                and PatientName fields

    --skip-if-set STR           Do nothing if a random sampling of dicoms in <scan_path>
                                already have their PatientName and PatientID fields
                                set to the given string

    --output-path DIR           Copy the results of anonymization into the given
                                directory rather than overwriting the original
                                archive

    -v, --verbose               Print messages for intermediate steps

"""

from docopt import docopt
import os.path
import sys
import glob
import random
import dicom as dcm
import time


VERBOSE = False

def error(msg, continue_exec=True):
    print("ERROR: " + msg)
    sys.stdout.flush()
    if not continue_exec:
        sys.exit(1)

def verbose(msg):
    if VERBOSE:
        print(msg)
        sys.stdout.flush()

def main():
    global VERBOSE
    arguments       = docopt(__doc__)
    scan_path       = arguments['<scan_path>']
    use_ID_field    = arguments['--patientID']
    given_ID        = arguments['--provideID']
    verify_ID       = arguments['--skip-if-set']
    set_path        = arguments['--output-path']
    VERBOSE         = arguments['--verbose']

    input_path = sanitize_path(scan_path)

    if not valid_scan_path(input_path):
        error("{} is not a valid scan_path. Check path and "\
              "folder structure.".format(input_path))
        sys.exit(1)

    if headers_already_anonymized(verify_ID, input_path):
        sys.exit()

    anon_ID = get_anonymized_ID(input_path, use_ID_field, given_ID)
    output_path = get_output_location(input_path, set_path)

    start_time = time.time()

    set_headers(input_path, output_path, anon_ID)

    verbose("Run time: {}".format(time.time() - start_time))

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

def valid_scan_path(scan_path):
    """
    Returns False if
        - <scan_path> is not a valid path
        - the scan at that location does not contain 1+ dicom
        - the path to the dicoms are not of the form:
                <scan_path>/<series_folders>/<dicoms>
    """
    if not os.path.isdir(scan_path):
        return False
    else:
        dicoms = glob.glob(os.path.join(scan_path, "*/*.dcm"))
        if len(dicoms) == 0:
            return False
    return True

def headers_already_anonymized(correct_ID, scan_folder):
    """
    Takes a random sample of one dicom per series folder and checks if both
    the PatientID and PatientName fields match the given ID.
    """
    if correct_ID is None:
        return False

    for dicom in get_random_sample(scan_folder):
        header = read_dicom(dicom)
        if header.PatientID != correct_ID or header.PatientName != correct_ID:
            return False

    verbose("{} already has ID {}".format(scan_folder, correct_ID))

    return True

def read_dicom(fname):
    try:
        dicom = dcm.read_file(fname)
        return dicom
    except:
        error("Failed to read {}. Check that it is"\
              " a valid dicom".format(fname), False)
    return None

def get_random_sample(scan_folder):
    """
    Returns a list of dicoms where each dicom is randomly selected from
    a series folder in scan_folder
    """
    sample = []

    for series_folder in glob.glob(os.path.join(scan_folder, '*')):
        if not os.path.isdir(series_folder):
            continue

        max_tries = 5
        tries = 0
        while tries < max_tries:
            scan = choose_item(series_folder)
            tries += 1
            if ".dcm" in scan:
                scan_path = os.path.join(series_folder, scan)
                sample.append(scan_path)
                break

    return sample

def choose_item(series_folder):
    """
    Chooses a random item from within the given folder
    """
    scans = os.listdir(series_folder)
    count = len(scans) - 1
    rand_num = random.randint(0, count)
    chosen_item = scans[rand_num]
    return chosen_item

def get_anonymized_ID(scan_path, use_pt_ID, given_ID):
    """
    Determines the correct ID to use for the given scan based on the options
    and current header.
    """
    if given_ID is not None:
        verbose("New ID for {} is {}".format(scan_path, given_ID))
        return given_ID

    for dicom in glob.glob(os.path.join(scan_path, '*/*.dcm')):
        header = read_dicom(dicom)
        if use_pt_ID:
            correct_ID = header.PatientID
        else:
            correct_ID = header.PatientName

        if correct_ID is not None:
            verbose("New ID for {} is {}".format(scan_path, correct_ID))
            return correct_ID

    # If correct_ID can't be determined by settings or headers
    # exit with error
    error("ID cannot be determined. Check that the needed"\
          "header field is not empty.", False)
    return None

def get_output_location(input_path, output_path):
    """
    Determines the location to save all the modified dicoms based on the settings
    """
    if output_path is None:
        path, folder_name = os.path.split(input_path)
        return path
    clean_output_path = sanitize_path(output_path)
    return clean_output_path

def set_headers(scan, output, ID):
    """
    Gives all dicoms in <scan> the given ID and leaves the result in <output>.

    This function retains the scan folder layout and will attempt to make the
    output directory if it doesn't already exist.
    """

    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except:
            error("Could not create output directory: {}".format(output), False)

    for dicom in glob.glob(os.path.join(scan, "*/*dcm")):
        header = read_dicom(dicom)
        header.PatientID = ID
        header.PatientName = ID
        dcm_path = get_dicom_output_path(dicom, output)
        header.save_as(dcm_path)

    verbose("Finished changing {} header fields to {}".format(scan, ID))

def get_dicom_output_path(dicom_path, output_path):
    series_path, dcm_name = os.path.split(dicom_path)
    scan_path, series_name = os.path.split(series_path)
    parent_path, scan_name = os.path.split(scan_path)

    if parent_path == output_path:
        return dicom_path

    new_path = os.path.join(output_path, scan_name, series_name)

    if not os.path.isdir(new_path):
        try:
            os.makedirs(new_path)
        except:
            error("Cannot make path "\
                  "{} to save {}".format(new_path, dcm_name), False)

    return os.path.join(new_path, dcm_name)

if __name__ == '__main__':
    main()
