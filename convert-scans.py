"""
Converts each scan stored under <input_dir> to the formats listed in <formats_csv>.
Results of conversion are stored in each scan's parent folder under a subdirectory
named after the format.

E.g. /input_dir/parent_folder/scan1/scan1_Fractional_Aniso converted to nifti
would be stored at /input_dir/parent_folder/nii/scan1_Fractional_Aniso.nii.gz

Output files will be named according to the scheme below, where tag is a short
and easily searchable representation of "series-description+:
    <scan_folder_name>_tag_series-number_series-description

Usage:
    convert-scans.py [options] <input_dir> <formats_csv>

Arguments:
    <input_dir>             The full path to the parent directory of all
                            scan folders. The scan folders must be unzipped
                            dicoms organized into subdirectories by series.

    <formats_csv>           The full path to the .csv file containing the
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

EXPORT TABLE FORMAT

    The <formats_csv> should contain a lookup table that supplies a pattern to
    match against the DICOM SeriesDescription header and corresponding tag name.
    Additionally, the export table should contain a column for each export
    filetype with "yes" if the series should be exported to that format.

    For example:

    pattern       tag     export_mnc  export_nii  export_nrrd  count
    Localiser     LOC     no          no          no           1
    Calibration   CAL     no          no          no           1
    Aniso         ANI     no          no          no           1
    HOS           HOS     no          no          no           1
    T1            T1      yes         yes         yes          1
    T2            T2      yes         yes         yes          1
    FLAIR         FLAIR   yes         yes         yes          1
    Resting       RES     no          yes         no           1
    Observe       OBS     no          yes         no           1
    Imitate       IMI     no          yes         no           1
    DTI-60        DTI-60  no          yes         yes          3
    DTI-33-b4500  b4500   no          yes         yes          1
    DTI-33-b3000  b3000   no          yes         yes          1
    DTI-33-b1000  b1000   no          yes         yes          1

"""

from docopt import docopt
import os
import sys
import glob
import tempfile
import shutil
import pandas as pd
import datman as dm
import re

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
    dm.utils.run(cmd, DRYRUN)

def main():
    global VERBOSE
    global DEBUG
    global DRYRUN
    arguments       = docopt(__doc__)
    input_dir       = arguments['<input_dir>']
    formats_csv     = arguments['<formats_csv>']
    blacklist_csv   = arguments['--blacklist']
    VERBOSE         = arguments['--verbose']
    DEBUG           = arguments['--debug']
    DRYRUN          = arguments['--dry-run']

    scan_list = find_all_scan_data(input_dir)

    format_df = make_dataframe(formats_csv)
    # Will cause program to exit if --blacklist set but file cant be parsed
    blacklist = get_blacklisted_series(blacklist_csv)

    for scan in scan_list:
        convert_needed_series(scan, format_df, blacklist)


def find_all_scan_data(input_dir):
    """
    Returns a list of absolute paths for all scan folders inside input_dir
    """
    scan_folders = []
    for sub_dir in glob.glob(os.path.join(input_dir, '*')):
        scan_folder = get_scan_folder_path(sub_dir)
        if scan_folder is not None:
            scan_folders.append(scan_folder)

    return scan_folders

def get_scan_folder_path(path):
    """
    If a dicom is found within a subfolder of 'path', returns the absolute
    path to the scan folder, otherwise returns None
    """
    if not os.path.isdir(path):
        return None

    for root, dirs, files in os.walk(path):
        for fname in files:
            if '.dcm' in fname:
                scan_folder, series_folder = os.path.split(root)
                return scan_folder

    return None

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

def get_formats_from_dataframe(dataframe):
    """
    Returns a list of the formats extracted from the 'export_' column headers
    """
    columns = dataframe.columns.values.tolist()
    formats = [c.split("_")[1] for c in columns if c.startswith("export_")]

    unknown_fmts = [fmt for fmt in formats if fmt not in exporters]
    if len(unknown_fmts) > 0:
        error("Unknown formats requested: {}." \
              " Skipping.".format(", ".join(unknown_fmts)))
        formats = list(set(formats) - set(unknown_fmts))

    return formats

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

def get_tag_map(format_df):
    patterns = format_df['pattern'].tolist()
    tags = format_df['tag'].tolist()
    return dict(zip(patterns, tags))

def convert_needed_series(scan_path, format_df, blacklist):
    """
    Finds series that need conversion and runs the necessary conversion
    programs.

    Adapted from datman's xnat-extract.py
    """
    scanid = os.path.basename(scan_path)
    output_folder, _ = os.path.split(scan_path)

    for series_folder, header in dm.utils.get_archive_headers(scan_path).items():
        # parse header
        description = str(header.get("SeriesDescription"))
        mangled_descr = mangle_description(description)
        series_num = str(header.get("SeriesNumber")).zfill(2)

        # Find tag and the row of the csv with export info related to it
        tag_map = get_tag_map(format_df)
        tag = dm.utils.guess_tag(mangled_descr, tag_map)

        if tag is None:
            verbose("No matching export patterns for "\
                    "{} with description: {}."\
                    " Skipping".format(series_folder, description))
            continue

        output_name = scanid + "_" + "_".join([tag, series_num, mangled_descr])

        if output_name in blacklist:
            debug("{} in blacklist. Skipping".format(output_name))
            continue

        formats = get_formats_from_dataframe(format_df)
        tag_row = format_df[format_df['tag'] == tag]

        for fmt in formats:
            if all(tag_row['export_' + fmt] == 'no'):
                debug("{}: export_{} set to 'no' for tag {}. Skipping".format(
                        output_name, fmt, tag))
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
        ext = dm.utils.get_extension(f)
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
