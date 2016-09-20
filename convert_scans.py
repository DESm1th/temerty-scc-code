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

    scan_dict = find_all_scan_data(input_dir)

    config = read_yaml_settings(proj_settings)

    # Will cause program to exit if --blacklist set but file cant be parsed
    blacklist = get_blacklisted_series(blacklist_csv)

    for scan_id in scan_dict.keys():
        convert_needed_series(scan_id, scan_dict[scan_id], config, blacklist)

def find_all_scan_data(input_dir):
    """
    Returns a dictionary of datman style subject ids mapped to an absolute
    path where the scan folder is stored.
    """
    all_scans = {}
    for sub_dir in glob.glob(os.path.join(input_dir, '*')):
        dicom = get_dicom(sub_dir)
        if dicom is None:
            # sub_dir is not a scan folder of dicoms
            continue

        scan_id = get_scan_id(dicom)

        if scan_id is None:
            continue

        scan_path = get_scan_folder_path(dicom)
        all_scans[scan_id] = scan_path

    return all_scans

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

def get_scan_id(dicom_path):
    """
    Uses the dicom header to assign a scan id of the format:
        <StudyDescription>_<InstitutionName>_<PatientName>_01_01

    The 01_01 fields are intended to be <timepoint>_<session#> but are not
    currently implemented. If <PatientName> has three underscore separated fields,
    the last two fields will be taken as timepoint and session instead. Currently
    this is the only way to adjust these values.
    """
    try:
        header = dcm.read_file(dicom_path)
    except:
        error("{} is not a readable dicom".format(dicom_path))
        return None

    scan_id = header.StudyDescription + "_" + header.InstitutionName + "_"

    patient_name = header.PatientName
    if len(patient_name.split("_")) == 3:
        scan_id += patient_name
    elif len(patient_name.split("_")) == 2:
        scan_id += patient_name + "_01"
    else:
        scan_id += patient_name + "_01_01"

    return scan_id

def get_scan_folder_path(dicom_path):
    """
    Returns the path to the scan folder that the given dicom belongs to
    """
    series_path, fname = os.path.split(dicom_path)
    scan_path, series_folder = os.path.split(series_path)
    return scan_path

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
