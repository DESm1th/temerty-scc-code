#!/usr/bin/env python
"""
Produces QC documents for each exam. Adapted from datman/bin/qc_html.py

Usage:
    qc_pages.py [options] <nii_folder> <project_settings> <output_dir>

Arguments:
    <nii_folder>            The path to a folder of nifti images for one participant

    <project_settings>      A .yml project settings file

    <output_dir>            The path to the desired output location.

Options:
    --datman-structure      Set if using the datman file structure organization
                            (if not set, it is assumed data is organized by
                            subject rather than pooled by format)
    --rewrite               Rewrite the html of an existing qc page
    --verbose               Be chatty
    --debug                 Be extra chatty

REQUIREMENTS

    This program requires the AFNI toolkit and FSL to be available, as well as NIFTI
    scans for each acquisition to be QC'd.

"""

import os
import glob
import sys
import logging
import numpy as np
import nibabel as nib
import datman_utils as dm_utils
import datman_scanid as dm_scanid
import subprocess as proc
from docopt import docopt
import re
import tempfile
import yaml
import pandas as pd
import img_utils as imgs

#########################################################
# Global constants.
REWRITE = False
LOGGER = None
DATMAN_ORG = False

#########################################################

def main():
    global REWRITE
    global LOGGER
    global DATMAN_ORG

    arguments   = docopt(__doc__)
    nii_folder  = arguments['<nii_folder>']
    yml_file    = arguments['<project_settings>']
    output_dir  = arguments['<output_dir>']
    DATMAN_ORG  = arguments['--datman-structure']
    verbose     = arguments['--verbose']
    debug       = arguments['--debug']
    REWRITE     = arguments['--rewrite']

    LOGGER = set_up_logging(verbose, debug)

    input_path = sanitize_path(nii_folder)
    output_path = sanitize_path(output_dir)
    yml_path = sanitize_path(yml_file)

    config = read_yaml_settings(yml_path)
    subject_id = get_subject_id(input_path)

    write_qc_page(input_path, output_path, subject_id, config)

def set_up_logging(verbose, debug):
    """
    Configures the logger for this run
    """
    logging.basicConfig(level=logging.WARN,
            format="[%(name)s] %(levelname)s: %(message)s")

    logger = logging.getLogger('qc_pages')

    if verbose:
        logger.setLevel(logging.INFO)

    if debug:
        logger.setLevel(logging.DEBUG)

    return logger

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

def read_yaml_settings(yml_file):
    try:
        with open(yml_file, 'r') as stream:
            pconfig = yaml.load(stream)
    except:
        logging.getLogger().error("{} cannot be read.".format(yml_file))
        sys.exit(-1)

    return pconfig

def get_subject_id(nii_path):
    """
    It is expected that the subject id is based on the datman convention:
        study_site_subject_timepoint

    As a result, the subject id is taken to be the first 4 fields of the
    underscore delimited nifti file name.
    """

    nifti_list = glob.glob(os.path.join(nii_path, "*.nii*"))

    if len(nifti_list) == 0:
        logging.getLogger().error("No nifti files found at {}".format(nii_path))
        sys.exit(1)

    # Grab the subject id from the first nifti found
    file_name_plus_ext = os.path.basename(nifti_list[0])
    file_name = remove_nifti_extension(file_name_plus_ext)
    fields = file_name.split("_")

    if len(fields) < 4:
        logging.getLogger().error("Cannot read subject id "\
                            "from {}".format(nifti_list[0]))
        sys.exit(1)

    subject_id = "_".join(fields[0:4])

    misnamed = [nifti for nifti in nifti_list if subject_id not in nifti]
    if len(misnamed) > 0:
        logging.getLogger().warn("{} contains misnamed nifti files. The "\
                    "following files should be renamed:\n{}".format(
                    nii_path, "\n".join(misnamed)))

    return subject_id

def remove_nifti_extension(file_name):
    return file_name.replace(".nii", "").replace(".gz", "")

def write_qc_page(input_path, output_path, subject_id, config):
    html_file = os.path.join(output_path, 'qc_{}.html'.format(subject_id))

    if os.path.exists(html_file) and not REWRITE:
        logging.getLogger().debug("{} exists, skipping.".format(html_file))
        return

    if REWRITE:
        try:
            os.remove(html_file)
        except:
            logging.getLogger().info("{} does not exist. Reconstructing html "\
                  "file from any images present.".format(html_file))

    with open(html_file, 'a') as qc_page:
        write_qc_page_header(qc_page, subject_id)
        exportinfo = create_exportinfo_dataframe(input_path, subject_id, config)
        write_qc_page_export_table(qc_page, exportinfo)
        write_tech_notes_link(qc_page, input_path, subject_id)
        write_qc_page_contents(qc_page, input_path, output_path, exportinfo, subject_id)

def write_qc_page_header(qc_page, subject_id):
    qc_page.write('<HTML><TITLE>{} qc</TITLE>\n'.format(subject_id))
    qc_page.write('<head>\n<style>\n'
                'body { font-family: futura,sans-serif;'
                '        text-align: center;}\n'
                'img {width:90%; \n'
                '   display: block\n;'
                '   margin-left: auto;\n'
                '   margin-right: auto }\n'
                'table { margin: 25px auto; \n'
                '        border-collapse: collapse;\n'
                '        text-align: left;\n'
                '        width: 90%; \n'
                '        border: 1px solid grey;\n'
                '        border-bottom: 2px solid black;} \n'
                'th {background: black;\n'
                '    color: white;\n'
                '    text-transform: uppercase;\n'
                '    padding: 10px;}\n'
                'td {border-top: thin solid;\n'
                '    border-bottom: thin solid;\n'
                '    padding: 10px;}\n'
                '</style></head>\n')

    qc_page.write('<h1> QC report for {} <h1/>'.format(subject_id))

def create_exportinfo_dataframe(input_path, subject_id, config):
    """
    Compares the contents of input_path to the export info in config
    and creates a dataframe from the results.
    """

    nifti_files = get_file_names(input_path)
    expected_contents = get_expected_scan_contents(config, subject_id)

    ### initialize the DataFrame
    cols = ['tag', 'File','bookmark', 'Note']
    exportinfo = pd.DataFrame(columns=cols)

    idx = 0
    for row in expected_contents:
        tag = row.keys()[0]
        expected_count = row[tag]['Count']
        tagged_files = get_tagged_files(tag, nifti_files)

        file_num = 0
        for nifti in tagged_files:
            bookmark = tag + str(file_num)
            notes = ''
            if file_num >= expected_count:
                notes = 'Repeated Scan'

            exportinfo.loc[idx] = [tag, nifti, bookmark, notes]
            idx += 1
            file_num += 1

        if file_num < expected_count:
            notes = "missing({})".format(expected_count - file_num)
            exportinfo.loc[idx] = [tag, '', '', notes]
            idx += 1

    # Make a note for all nifti files that don't match the export info
    expected_files = get_all_expected_files(exportinfo)
    other_scans = list(set(nifti_files) - set(expected_files))
    for scan in other_scans:
        exportinfo.loc[idx] = ['unknown', scan, '', 'extra scan']
        idx += 1

    return(exportinfo)

def get_file_names(input_path):
    names = []
    for nifti in glob.glob(os.path.join(input_path, '*.nii*')):
        names.append(os.path.basename(nifti))
    return names

def get_expected_scan_contents(config, subject_id):
    """
    This function will attempt to match the 'site' field from subject_id
    to a site in config to find the 'ExportInfo' to use for this scan.

    If no match can be made, this function will exit the program.
    """

    expected_contents = None

    for site_config in config['Sites']:
        site = site_config.keys()[0]
        site_tag = "_" + site + "_"
        if site_tag in subject_id:
            expected_contents = site_config[site]['ExportInfo']

    if expected_contents is None:
        logging.getLogger().error("ID {} does not belong to any site in"\
                    " the config file".format(subject_id))
        sys.exit(1)

    return expected_contents

def get_tagged_files(tag_pattern, nifti_files):
    tag = "_" + tag_pattern + "_"
    tagged_files = [nifti for nifti in nifti_files if tag in nifti]
    tagged_files.sort()
    return tagged_files

def get_all_expected_files(exportinfo):
    exportinfoFiles = exportinfo.File.tolist()
    PDT2scans = [k for k in exportinfoFiles if '_PDT2_' in k]
    if len(PDT2scans) > 0:
        for PDT2scan in PDT2scans:
            exportinfoFiles.append(PDT2scan.replace('_PDT2_','_T2_'))
            exportinfoFiles.append(PDT2scan.replace('_PDT2_','_PD_'))

    return exportinfoFiles

def write_qc_page_export_table(qc_html, exportinfo):
    ##write table header
    qc_html.write('<table>'
                '<tr><th>Tag</th>'
                '<th>File</th>'
                '<th>Notes</th></tr>')

    ## for each row write the table data
    for row in range(0,len(exportinfo)):
        qc_html.write('<tr><td>{}</td>'.format(exportinfo.loc[row,'tag'])) ## table new row
        qc_html.write('<td><a href="#{}">{}</a></td>'.format(exportinfo.loc[row,'bookmark'],exportinfo.loc[row,'File']))
        qc_html.write('<td><font color="#FF0000">{}</font></td></tr>'.format(exportinfo.loc[row,'Note'])) ## table new row

    ##end table
    qc_html.write('</table>\n')

def write_tech_notes_link(qc_page, input_path, subject_id):

    resources_path = os.path.join(input_path, "..")

    if DATMAN_ORG:
        resources_path = os.path.join(resources_path, "../RESOURCES/{}*".format(subject_id))

    resources_path = os.path.abspath(resources_path)
    tech_notes_path = find_tech_notes(resources_path)

    if tech_notes_path is None:
        qc_page.write('<p>Tech Notes not found</p>\n')
        return

    qc_page.write('<a href="'+ tech_notes_path + '" >')
    qc_page.write('Click Here to open Tech Notes')
    qc_page.write('</a><br>\n')
    return

def find_tech_notes(path):
    """
    Search the file tree rooted at path for the tech notes pdf
    """
    for root, dirs, files in os.walk(glob.glob(path)[0]):
        for fname in files:
            if ".pdf" in fname:
                return os.path.join(root, fname)
    return None

def write_qc_page_contents(qc_page, input_path, output_path, exportinfo, subject_id):

    # load up any header/bvec check log files for the subject
    logs_path = os.path.join(input_path, 'logs')
    header_check_log = get_log_contents(logs_path, 'dm-check-headers-{}*'.format(subject_id))
    bvecs_check_log = get_log_contents(logs_path, 'dm-check-bvecs-{}*'.format(subject_id))

    for idx in range(0,len(exportinfo)):
        file_name = exportinfo.loc[idx,'File']
        if file_name == '':
            continue
        curr_file = os.path.join(input_path, file_name)
        logging.getLogger().info("QC scan {}".format(curr_file))
        ident, tag, series, description = dm_scanid.parse_filename(curr_file)
        qc_page.write('<h2 id="{}">{}</h2>\n'.format(exportinfo.loc[idx,'bookmark'], file_name))

        if tag not in QC_HANDLERS:
            logging.getLogger().info("MSG: No QC tag {} for scan {}. "\
                                     "Skipping.".format(tag, curr_file))
            continue

        if header_check_log and tag != 'PDT2':
            add_header_checks(curr_file, qc_page, header_check_log)
        if bvecs_check_log:
            add_bvec_checks(curr_file, qc_page, bvecs_check_log)

        if not REWRITE:
            QC_HANDLERS[tag](curr_file, output_path, qc_page)
        else:
            add_old_image(curr_file, output_path, qc_page, tag)

        qc_page.write('<br>')

def get_log_contents(log_path, file_pattern):
    logs = glob.glob(os.path.join(log_path, file_pattern))

    log_contents = []
    for log_file in logs:
        log_contents += open(log_file).readlines()

    return log_contents

def add_header_checks(file_path, qc_page, log_data):
    filestem = os.path.basename(file_path).replace(dm_utils.get_extension(file_path),'')
    lines = [re.sub('^.*?: *','',line) for line in log_data if filestem in line]
    if not lines:
        return

    qc_page.write('<h3> {} header differences </h3>\n<table>'.format(filestem))
    for l in lines:
        fields = l.split(',')
        qc_page.write('<tr>')
        for item in fields:
            qc_page.write('<td align="center">{}</td>'.format(item))
        qc_page.write('</tr>')
    qc_page.write('</table>\n')

def add_bvec_checks(fpath, qchtml, logdata):
    filestem = os.path.basename(fpath).replace(dm_utils.get_extension(fpath),'')
    lines = [re.sub('^.*'+filestem,'',line) for line in logdata if filestem in line]
    if not lines:
        return

    qchtml.write('<h3> {} bvec/bval differences </h3>\n<table>'.format(filestem))
    for l in lines:
        fields = l.split(',')
        qchtml.write('<tr>')
        for item in fields:
            qchtml.write('<td align="center">{}</td>'.format(item))
        qchtml.write('</tr>')
    qchtml.write('</table>\n')

def add_old_image(fpath, qcpath, qchtml, tag):
    fname = nifti_basename(fpath)
    fname = os.path.join(qcpath, fname)

    if os.path.exists(fname + '.png'):
        add_pic_to_html(qchtml, fname + '.png')
        return
    elif os.path.exists(fname + '_BOLD.png'):
        add_pic_to_html(qchtml, fname + '_BOLD.png')
        add_pic_to_html(qchtml, fname + '_fmriplots.png')
        add_pic_to_html(qchtml, fname + '_SNR.png')
        add_pic_to_html(qchtml, fname + '_Spikes.png')
    elif os.path.exists(fname + '_B0.png'):
        add_pic_to_html(qchtml, fname + '_B0.png')
        add_pic_to_html(qchtml, fname + '_dti4d.png')
        add_pic_to_html(qchtml, fname + '_spikes.png')
    elif tag == 'PDT2':
        add_pic_to_html(qchtml, fname.replace('_PDT2_','_PD_') + '.png')
        add_pic_to_html(qchtml, fname.replace('_PDT2_', '_T2_') + '.png')

###############################################################################
# PIPELINES
# Taken from datman/bin/qc_html.py

def ignore(fpath, qcpath, qchtml):
    pass

def fmri_qc(fpath, qcpath, qchtml):
    """
    This takes an input image, motion corrects, and generates a brain mask.
    It then calculates a signal to noise ratio map and framewise displacement
    plot for the file.
    """
    # if the number of TRs is too little, we skip the pipeline
    ntrs = check_n_trs(fpath)

    if ntrs < 20:
        return

    filename = os.path.basename(fpath)
    filestem = nifti_basename(fpath)
    tmpdir = tempfile.mkdtemp(prefix='qc-')

    error_message = "Command {} not found. Check that AFNI is installed."

    run('3dvolreg \
         -prefix {t}/mcorr.nii.gz \
         -twopass -twoblur 3 -Fourier \
         -1Dfile {t}/motion.1D {f}'.format(t=tmpdir, f=fpath),
         error_message.format('3dvolreg'))
    run('3dTstat -prefix {t}/mean.nii.gz {t}/mcorr.nii.gz'.format(t=tmpdir),
         error_message.format('3dTstat'))
    run('3dAutomask \
         -prefix {t}/mask.nii.gz \
         -clfrac 0.5 -peels 3 {t}/mean.nii.gz'.format(t=tmpdir),
         error_message.format('3dAutomask'))
    run('3dTstat -prefix {t}/std.nii.gz  -stdev {t}/mcorr.nii.gz'.format(t=tmpdir),
         error_message.format('3dTstat'))
    run("""3dcalc \
           -prefix {t}/sfnr.nii.gz \
           -a {t}/mean.nii.gz -b {t}/std.nii.gz -expr 'a/b'""".format(t=tmpdir),
        error_message.format('3dcalc'))


    # output BOLD-contrast qc-pic
    BOLDpic = os.path.join(qcpath, filestem + '_BOLD.png')
    imgs.montage(fpath, 'BOLD-contrast', filename, BOLDpic, maxval=0.75)
    add_pic_to_html(qchtml, BOLDpic)

    # output fMRI plots
    fMRIplotspic = os.path.join(qcpath, filestem + '_fmriplots.png')
    imgs.fmri_plots('{t}/mcorr.nii.gz'.format(t=tmpdir),
                     '{t}/mask.nii.gz'.format(t=tmpdir),
                     '{t}/motion.1D'.format(t=tmpdir), filename, fMRIplotspic)
    add_pic_to_html(qchtml, fMRIplotspic)

    SNRpic = os.path.join(qcpath,filestem + '_SNR.png')
    imgs.montage('{t}/sfnr.nii.gz'.format(t=tmpdir),
                  'SFNR', filename, SNRpic, cmaptype='hot', maxval=0.75)
    add_pic_to_html(qchtml, SNRpic)


    Spikespic = os.path.join(qcpath,filestem + '_Spikes.png')
    imgs.find_epi_spikes(fpath, filename, Spikespic)
    add_pic_to_html(qchtml, Spikespic)

    run('rm -r {}'.format(tmpdir))

def check_n_trs(fpath):
    """
    Returns the number of TRs for an input file. If the file is 3D, we also
    return 1.
    """
    data = nib.load(fpath)

    try:
        ntrs = data.shape[3]
    except:
        ntrs = 1

    return ntrs

def rest_qc(fpath, qcpath, qchtml):
    """
    This takes an input image, motion corrects, and generates a brain mask.
    It then calculates a signal to noise ratio map and framewise displacement
    plot for the file.
    """
    fmri_qc(fpath, qcpath, qchtml)

def pdt2_qc(fpath, qcpath, qchtml):
    ## split PD and T2 image
    pdpath = fpath.replace('_PDT2_','_PD_')
    t2path = fpath.replace('_PDT2_','_T2_')
    pd_qc(pdpath, qcpath, qchtml)
    t2_qc(t2path, qcpath, qchtml)

def t1_qc(fpath, qcpath, qchtml):
    pic=os.path.join(qcpath, nifti_basename(fpath) + '.png')
    fslslicer_pic(fpath,pic,5,1600)
    add_pic_to_html(qchtml, pic)

def pd_qc(fpath, qcpath, qchtml):
    pic=os.path.join(qcpath, nifti_basename(fpath) + '.png')
    fslslicer_pic(fpath,pic,2,1600)
    add_pic_to_html(qchtml, pic)

def t2_qc(fpath, qcpath, qchtml):
    pic=os.path.join(qcpath, nifti_basename(fpath) + '.png')
    fslslicer_pic(fpath,pic,2,1600)
    add_pic_to_html(qchtml, pic)

def flair_qc(fpath,qcpath, qchtml):
    pic=os.path.join(qcpath, nifti_basename(fpath) + '.png')
    fslslicer_pic(fpath,pic,2,1600)
    add_pic_to_html(qchtml, pic)

def dti_qc(fpath, qcpath, qchtml):
    """
    Runs the QC pipeline on the DTI inputs. We use the BVEC (not BVAL)
    file to find B0 images (in some scans, mid-sequence B0s are coded
    as non-B0s for some reason, so the 0-direction locations in BVEC
    seem to be the safer choice).
    """
    filename = os.path.basename(fpath)
    filestem = nifti_basename(fpath)
    directory = os.path.dirname(fpath)
    bvecfile = fpath[:-len(dm_utils.get_extension(fpath))] + ".bvec"
    bvalfile = fpath[:-len(dm_utils.get_extension(fpath))] + ".bval"

    # load in bvec file
    logging.getLogger().debug("fpath = {}, bvec = {}".format(fpath, bvecfile))

    if not os.path.exists(bvecfile):
        logging.getLogger().warn("Expected bvec file not found: {}. Skipping".format(bvecfile))
        return

    bvec = np.genfromtxt(bvecfile)
    bvec = np.sum(bvec, axis=0)

    B0pic = os.path.join(qcpath,filestem + '_B0.png')
    imgs.montage(fpath, 'B0-contrast', filename, B0pic, maxval=0.25)
    add_pic_to_html(qchtml, B0pic)

    dti4dpic = os.path.join(qcpath,filestem + '_dti4d.png')
    imgs.montage(fpath, 'DTI Directions', filename, dti4dpic, mode='4d', maxval=0.25)
    add_pic_to_html(qchtml, dti4dpic)

    spikespic = os.path.join(qcpath, filestem + '_spikes.png')
    imgs.find_epi_spikes(fpath, filename, spikespic, bvec=bvec)
    add_pic_to_html(qchtml, spikespic)

def fslslicer_pic(fpath,pic,slicergap,picwidth):
    """
    Uses FSL's slicer function to generate a pretty montage png from a nifti file
    Then adds a link to that png in the qcthml

    Usage:
        add_slicer_pic(fpath,slicergap,picwidth,qchtml)

        fpath       -- submitted image file name
        slicergap   -- int of "gap" between slices in Montage
        picwidth    -- width (in pixels) of output image
        pic         -- fullpath to for output image
    """
    run("slicer {} -S {} {} {}".format(fpath,slicergap,picwidth,pic))

def add_pic_to_html(qchtml, pic):
    '''
    Adds a pic to an html page with this handler "qchtml"
    '''
    relpath = os.path.relpath(pic,os.path.dirname(qchtml.name))
    qchtml.write('<a href="'+ relpath + '" >')
    qchtml.write('<img src="' + relpath + '" > ')
    qchtml.write('</a><br>\n')
    return qchtml

def nifti_basename(fpath):
    """
    return basename without .nii.gz extension
    """
    basefpath = os.path.basename(fpath)
    stem = remove_nifti_extension(basefpath)
    return(stem)

def makedirs(path):
    logging.getLogger().debug("makedirs: {}".format(path))
    os.makedirs(path)

def run(cmd, error_message=None):

    logging.getLogger().debug("exec: {}".format(cmd))
    p = proc.Popen(cmd, shell=True, stdout=proc.PIPE, stderr=proc.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:

        if error_message is None:
            error_message = "Error {} while executing: {}".format(p.returncode, cmd)

        logging.getLogger().error(error_message)
        out and logging.getLogger().error("stdout: \n>\t{}".format(out.replace('\n','\n>\t')))
        err and logging.getLogger().error("stderr: \n>\t{}".format(err.replace('\n','\n>\t')))

# map from tag to QC function
QC_HANDLERS = {
        "T1"            : t1_qc,
        "T2"            : t2_qc,
        "PD"            : pd_qc,
        "PDT2"          : pdt2_qc,
        "FLAIR"         : flair_qc,
        "FMAP"          : ignore,
        "FMAP-6.5"      : ignore,
        "FMAP-8.5"      : ignore,
        "RST"           : rest_qc,
        "EPI"           : fmri_qc,
        "SPRL"          : rest_qc,
        "OBS"           : fmri_qc,
        "IMI"           : fmri_qc,
        "NBK"           : fmri_qc,
        "EMP"           : fmri_qc,
        "VN-SPRL"       : fmri_qc,
        "SID"           : fmri_qc,
        "MID"           : fmri_qc,
        "DTI"           : dti_qc,
        "DTI21"         : dti_qc,
        "DTI22"         : dti_qc,
        "DTI23"         : dti_qc,
        "DTI60-29-1000" : dti_qc,
        "DTI60-20-1000" : dti_qc,
        "DTI60-1000"    : dti_qc,
        "DTI60-b1000"   : dti_qc,
        "DTI33-1000"    : dti_qc,
        "DTI33-b1000"   : dti_qc,
        "DTI33-3000"    : dti_qc,
        "DTI33-b3000"   : dti_qc,
        "DTI33-4500"    : dti_qc,
        "DTI33-b4500"   : dti_qc
}

if __name__ == "__main__":
    main()
