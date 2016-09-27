#!/usr/bin/env python
"""
Converts the scans stored in <input_dir> to the formats listed in
<project_settings>. Results of conversion are stored in each scan's parent
folder under a subdirectory named after the format.

E.g. /input_dir/parent_folder/scan1/scan1_Fractional_Aniso converted to nifti
would be stored at /input_dir/parent_folder/nii/scan1_Fractional_Aniso.nii.gz

Output files will be named according to the scheme below, where tag is a short
and easily searchable representation of "series-description":
    <scan_folder_name>_tag_series-number_series-description

Usage:
    convert_scans.py [options] <input_dir> <project_settings>

Arguments:
    <input_dir>             The full path to the parent directory of all
                            scan folders. The scan folders must be unzipped
                            dicoms organized into subdirectories by series.

    <project_settings>      The full path to the .yaml/.yml file containing the
                            table of series and formats to convert to. See below
                            for an example of the proper format.

Options:
    --blacklist FILE        The full path to a .csv file of blacklisted series
                            to ignore
    -v, --verbose           Show intermediate steps
    --debug                 Show debug messages
    -n, --dry-run           Do nothing


SCAN FOLDER FORMAT

    It is assumed that the scan data is organized in the following structure
    (directory names do not matter): scan_folder/series_folders/dicoms.

       E.g.

        exam_folder_sub_id
            /sub_id_SagT1Bravo
                *.dcm
            /sub_id_AxEPI-NBack
                *.dcm
            ...

REQUIREMENTS

    This program requires that the minc-toolkit and slicer be installed.

"""

from docopt import docopt
import os
import sys
import glob
import tempfile
import shutil
import re
import pandas as pd
import datman_utils as dm_utils
import dicom as dcm
import yaml


VERBOSE = False
DEBUG = False
DRYRUN = False

def error(msg, continue_exec=True):
    print("ERROR: " + msg)
    sys.stdout.flush()
    if not continue_exec:
        sys.exit(1)

def verbose(msg):
    if VERBOSE:
        print(msg)
        sys.stdout.flush()

def debug(msg):
    if DEBUG:
        print("DEBUG: " + msg)
        sys.stdout.flush()

def run(cmd):
    debug("exec: {}".format(cmd))
    dm_utils.run(cmd, DRYRUN)

def main():
    global VERBOSE
    global DEBUG
    global DRYRUN
    arguments       = docopt(__doc__)
    input_dir       = arguments['<input_dir>']
    proj_settings   = arguments['<project_settings>']
    blacklist_csv   = arguments['--blacklist']
    VERBOSE         = arguments['--verbose']
    DEBUG           = arguments['--debug']
    DRYRUN          = arguments['--dry-run']

    input_dir = sanitize_path(input_dir)
    proj_settings = sanitize_path(proj_settings)

    scans = get_scans(input_dir)
    scan_dict = assign_ids(scans)

    config = read_yaml_settings(proj_settings)

    # Will cause program to exit if --blacklist set but file cant be parsed
    blacklist = get_blacklisted_series(blacklist_csv)

    for scan_id in scan_dict.keys():
        convert_needed_series(scan_id, scan_dict[scan_id], config, blacklist)

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

def get_scans(path):
    if not os.path.isdir(path):
        return []

    if is_scan(path):
        return [path]

    scans = []
    for folder in glob.glob(os.path.join(path, '*')):
        scans.extend(get_scans(folder))

    return scans

def is_scan(path):
    for dicom in glob.glob(os.path.join(path, '*/*.dcm')):
        # If the folder structure matches and a dicom can be found
        # this is a scan
        return True
    return False

def assign_ids(scans):
    id_map = {}
    for scan in scans:
        dicom = get_dicom(scan)
        guessed_id = guess_scan_id(dicom)
        scan_id = resolve_id_conflict(guessed_id, dicom, id_map)
        if scan_id is None:
            error("Cannot assign ID to {} because a session collected at a"\
                  "later date has been assigned an earlier session number during"\
                  "a previous export. Please change the session number manually or delete"\
                  "all exports for this subject and rerun".format(scan))
            continue
        id_map[scan_id] = scan
    return id_map

def get_dicom(path):
    """
    If <path> contains at least one dicom in its subdirectories returns the path
    to the first one found. Otherwise returns None.
    """
    if not os.path.isdir(path):
        return None

    for root, dirs, files in os.walk(path):
        for fname in files:
            if '.dcm' in fname:
                return os.path.join(root, fname)
    return None

def guess_scan_id(dicom_path):
    """
    Uses the dicom header to assign a scan id of the format:
        <StudyDescription>_<InstitutionName>_<PatientName>_<timepoint>_<session>

    Timepoint: If PatientName is of the form someid_somenum the _somenum will
    be taken as timepoint. Otherwise, timepoint will be assigned to be _01

    Session: Assigned to be 01, but can be incremented with resolve_id_conflict()

    Note: If PatientName contains more than two underscore separated fields
    the fields will be merged and the default time point of _01 will be assigned
    even if the last field is numeric. e.g. ABC_DEF_008 will become ACBDEF008_01_01
    """
    try:
        header = dcm.read_file(dicom_path)
    except:
        error("{} is not a readable dicom".format(dicom_path))
        return None

    scan_id = header.StudyDescription + "_" + header.InstitutionName + "_"

    patient_name = header.PatientName
    name_fields = patient_name.split('_')
    last_index = len(name_fields) - 1

    if len(name_fields) == 2 and name_fields[last_index].isdigit():
        time_and_session = "_" + name_fields.pop(last_index) + "_01"
    else:
        time_and_session = "_01_01"

    scan_id += ''.join(name_fields) + time_and_session

    return scan_id

def resolve_id_conflict(scan_id, current_dicom, id_map):
    """
    If a scan in id_map already has scan_id, increments the
    session number of the scan that has a later StudyDate.
    Conflicts between this new scan and the id_map are then resolved until
    all scans in id_map have a unique id.
    """
    if scan_id not in id_map.keys():
        return scan_id

    other_scan = id_map[scan_id]
    other_dicom = get_dicom(other_scan)
    new_id = increment_session(scan_id)

    if session_date(other_dicom) > session_date(current_dicom):
        if previously_exported(other_scan, scan_id):
            return None
        new_other_id = resolve_id_conflict(new_id, other_dicom, id_map)
        id_map[new_other_id] = id_map.pop(scan_id)
        return scan_id
    else:
        return resolve_id_conflict(new_id, current_dicom, id_map)

def increment_session(scan_id):
    id_fields = scan_id.split("_")
    last_field = len(id_fields) - 1
    session_num = int(id_fields[last_field])
    session_num += 1
    id_fields[last_field] = str(session_num).zfill(2)
    return "_".join(id_fields)

def session_date(dicom):
    header = dcm.read_file(dicom)
    session_date = header.StudyDate
    return session_date

def previously_exported(scan, scan_id):
    parent_folder = os.path.split(scan)[0]

    for fmt in exporters.keys():
        export_dir = os.path.join(parent_folder, "{}".format(fmt))
        for exported_file in glob.glob(os.path.join(export_dir, "*{}*".format(scan_id))):
            # An item from scan has been exported with this ID for one or more format.
            return True
    return False

def read_yaml_settings(yaml_file):
    try:
        with open(yaml_file, 'r') as stream:
            yaml_settings = yaml.load(stream)
    except:
        error("{} cannot be read".format(yaml_file), False)

    return yaml_settings

def make_dataframe(csv):
    """
    Attempt to parse a csv into a pandas dataframe and exits the program
    if this fails.
    """
    try:
        dataframe = pd.read_table(csv, sep="\s+|,", engine="python")
    except:
        error("{} does not exist or cannot be read".format(csv),
               continue_exec=False)
    return dataframe

def get_blacklisted_series(blacklist_csv):
    """
    Parses the provided csv into a list of blacklisted series. If --blacklist
    is not set it returns the empty list
    """
    if blacklist_csv is None:
        return []

    blacklist = make_dataframe(blacklist_csv)
    series_list = blacklist.columns.tolist()[0]
    blacklisted_series = blacklist[series_list].values.tolist()

    return blacklisted_series

def convert_needed_series(scan_id, scan_path, project_config, blacklist):
    """
    Finds series that need conversion and runs the necessary conversion
    programs.

    Adapted from datman's xnat-extract.py
    """

    output_folder, _ = os.path.split(scan_path)
    export_info = get_export_info(scan_id, project_config)

    for series_folder, header in dm_utils.get_archive_headers(scan_path).items():
        description = str(header.get("SeriesDescription"))
        mangled_descr = mangle_description(description)
        series_num = str(header.get("SeriesNumber")).zfill(2)

        series_info = find_series_info(mangled_descr, export_info)
        if series_info is None:
            verbose("No matching export patterns for "\
                    "{} with description: {}."\
                    " Skipping".format(series_folder, description))
            continue

        tag = series_info.keys()[0]
        output_name = scan_id + "_" + "_".join([tag, series_num, mangled_descr])

        if output_name in blacklist:
            debug("{} in blacklist. Skipping".format(output_name))
            continue

        formats = series_info[tag]['Formats']
        for fmt in formats:
            if fmt not in exporters.keys():
                error("Unknown format {} requested for {}. Skipping.".format(fmt, output_name))
                continue

            output_dir = os.path.join(output_folder, fmt)
            if not os.path.exists(output_dir) and not DRYRUN:
                os.makedirs(output_dir)

            exporters[fmt](series_folder, output_dir, output_name)


def mangle_description(description):
    """
    Replaces runs of non-alphanumeric characters with a single dash
    and ensures the result does not begin or end with a dash
    """
    if description != "":
        # Replace each run of non-alphanumeric chars with a single dash
        mangled = re.sub(r"[^0-9a-zA-Z]+", '-', description)

        if re.match(r"[\W]", mangled[-1]):
            #strip dash from end
            mangled = mangled[:-1]

        if re.match(r"[\W]", mangled[0]):
            #strip dash from beginning
            mangled = mangled[1:]

        return mangled
    return ""

def get_export_info(scan_id, project_config):
    """
    Returns the ExportInfo for the site matching the given scan_id. May
    return None if no matching site is found.
    """
    export_info = None
    for site_config in project_config['Sites']:
        site = site_config.keys()[0]
        site_tag = "_" + site + "_"
        if site_tag in scan_id:
            export_info = site_config[site]['ExportInfo']
            break
    return export_info

def find_series_info(mangled_description, export_info):
    """
    Finds the line, if any, in export_info that applies to the given series based
    on a match to the mangled_description from the header.
    """
    series_info = None

    for line in export_info:
        tag = line.keys()[0]
        tag_info = line[tag]
        pattern = tag_info['Pattern']

        if isinstance(pattern, list):
            for string_descr in pattern:
                if string_descr in mangled_description:
                    series_info = line
                    break
        else:
            if pattern in mangled_description:
                series_info = line
                break

    return series_info

def export_mnc_command(seriesdir,outputdir,file_name):
    """
    Converts a DICOM series to MINC format

    Taken from datman's xnat-extract.py
    """
    outputfile = os.path.join(outputdir,file_name) + ".mnc"

    if os.path.exists(outputfile):
        debug("{}: output {} exists. skipping.".format(
            seriesdir, outputfile))
        return

    verbose("Exporting series {} to {}".format(seriesdir, outputfile))
    cmd = 'dcm2mnc -fname {} -dname "" {}/* {}'.format(
            file_name,seriesdir,outputdir)
    run(cmd)

def export_nii_command(seriesdir,outputdir,file_name):
    """
    Converts a DICOM series to NifTi format

    Taken from datman's xnat-extract.py
    """
    outputfile = os.path.join(outputdir,file_name) + ".nii.gz"

    if os.path.exists(outputfile):
        debug("{}: output {} exists. skipping.".format(
            seriesdir, outputfile))
        return

    verbose("Exporting series {} to {}".format(seriesdir, outputfile))

    # convert into tempdir
    tmpdir = tempfile.mkdtemp()
    run('dcm2nii -x n -g y  -o {} {}'.format(tmpdir,seriesdir))

    # move nii in tempdir to proper location
    for f in glob.glob("{}/*".format(tmpdir)):
        bn = os.path.basename(f)
        ext = dm_utils.get_extension(f)
        if bn.startswith("o") or bn.startswith("co"):
            continue
        else:
            run("mv {} {}/{}{}".format(f, outputdir, file_name, ext))
    shutil.rmtree(tmpdir)

def export_nrrd_command(seriesdir,outputdir,file_name):
    """
    Converts a DICOM series to NRRD format

    Taken from datman's xnat-extract.py
    """
    outputfile = os.path.join(outputdir,file_name) + ".nrrd"

    if os.path.exists(outputfile):
        debug("{}: output {} exists. skipping.".format(
            seriesdir, outputfile))
        return

    verbose("Exporting series {} to {}".format(seriesdir, outputfile))

    cmd = 'DWIConvert -i {} --conversionMode DicomToNrrd -o {}.nrrd ' \
          '--outputDirectory {}'.format(seriesdir,file_name,outputdir)

    run(cmd)

exporters = {
    "mnc" : export_mnc_command,
    "nii" : export_nii_command,
    "nrrd" : export_nrrd_command,
}

if __name__ == '__main__':
    main()
