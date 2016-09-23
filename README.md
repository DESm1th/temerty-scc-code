#Set up:

  1. Clone this repo
  2. Create a private module for it
  3. Ensure the some version of the following modules are loaded before running
     the pipeline (in addition to use.own and the qc-pipeline module):
     * Python 2.X (PYTHON/2.7.8-anaconda-2.1.0 is the one I used on the scc)
     * AFNI
     * FSL
     * minc-toolkit
     * mricron
     * slicer (Only if .nrrd file format is required)
  4. Ensure the following packages are installed. "pip install --user <package_name>"
     will install each locally and automatically for the version of python you have
     loaded.
     * docopt
     * pydicom
     * nibabel
  5. For each project to be managed, create a directory for logs and metadata.
     Each needed metadata configuration file has a template in the 'Config Templates'
     section below.

# Private Modules
  To use private modules on the SCC you must first load the use.own module.
  The first time you do this it creates a privatemodules folder in your home
  directory. In this directory you can create your own modules with the template
  below.

  The name you save the module file as determines what you'll have to type to load
  the module and the paths on each 'prepend-path' line will have to be adjusted
  to where you saved the clone repositories.


   qc-pipeline module template

     #%Module
     proc ModulesHelp { } {
     	puts stderr "This module loads the Temerty versions of some datman scripts"
     }

     # Load dependencies
     module load PYTHON/2.7.8-anaconda-2.1.0
     module load AFNI/AFNI_MAY_2014
     module load FSL/fsl_5.0.9
     module load MINC/minc-toolkit-1.0.07
     module load MRICRON/lx64

     prepend-path PATH	~/current/temerty-scc-code

     prepend-path PYTHONPATH	~/current/temerty-scc-code


# Config Templates

  project-settings.yml:

  This file will be unreadable if there are any tabbed white space. It's a good idea
  to use a text editor that can automatically convert tabs to spaces (like atom) to
  edit it to avoid weird errors.

  The first column after 'ExportInfo' is a tag that will be added to file names to
  make it easy to locate members of that series. The pattern column is a regex or
  simple string that will match the SeriesDescription field of the dicom headers.
  The site (CAMH in this case) is matched against the InstitutionName field of
  the header to determine which export info to use in the case of multiple scan sites
  for the given project.

    ###############
    # NEUR2MR Project Settings
    ###############

    STUDYNAME : NEUR2MR

    Sites:
      - CAMH :
          ## The data to expect from this project
          # Pattern: the regex pattern used to identify this scan from the dicom headers
          # Formats: the list of formats to convert matching series to
          # Count: the number of scans of this type to expect in a session
          ExportInfo:
            - LOC:           { Pattern: '3PlaneLoc',        Formats: [],              Count: 1}
            - T1:            { Pattern: 'T1',               Formats: [nii],           Count: 1}
            - CAL:           { Pattern: 'Calibration',      Formats: [],              Count: 1}
            - RST:           { Pattern: 'Resting',          Formats: [nii],           Count: 3}
            - GNG:           { Pattern: 'GoNoGo',           Formats: [nii],           Count: 1}
            - NBK:           { Pattern: 'Back',             Formats: [nii],           Count: 1}
            - DTI60-1000:    { Pattern: 'DTI-60plus5',      Formats: [nii],           Count: 1}
            - T2:            { Pattern: 'T2',               Formats: [nii],           Count: 1}
            - FMAP-6.5:      { Pattern: 'TE65',             Formats: [nii],           Count: 1}
            - FMAP-8.5:      { Pattern: 'TE85',             Formats: [nii],           Count: 1}
            - PROC:          { Pattern: 'Processed',        Formats: [],              Count: 1}
            - ANI:           { Pattern: 'Fractional-Aniso', Formats: [nii],           Count: 1}

  blacklist.csv:

  Another input to convert_scans.py. Not super useful unless you intend to use
  the rest of the datman pipeline. The series column should contain the datman
  style name of a series that has been blacklist. Reason can be anything, as
  long as it doesn't contain spaces (this breaks the parsing)

      series									                                            reason
      STUDYNAME_SITENAME_SUBJECTID_TIMEPOINT_SESSION_SERIESDESCRIPTION    just-because

# Naming Conventions

  The outputs of this qc pipeline use the datman naming conventions. All IDs
  created for participant data will be of the form the form
  STUDY_SITE_SUBJECTID_TIMEPOINT_SESSION.

  Most of this information is taken from the dicom headers. To set the timepoint
  for some scan, the PatientName field of the dicom headers should end in an
  underscore followed by the number of the timepoint. If this field isn't present
  at the end of the PatientName, then the timepoint will be set to 01 by default.

  Session number is set when convert_scans.py is run. It finds all dicoms with
  an ID that would match it for STUDY_SITE_SUBJECTID_TIMEPOINT and then calculates
  session numbers for each scan in the set by comparing their dates.

  __WARNING:__ If your scans are being uploaded and converted to other formats out of
  order convert_scans.py may ignore some of your new data. For example, if a participant
  with two sessions has the second session uploaded, unzipped, and converted to nifti
  and then later on the first scan is found the first scan will be ignored until
  the second one (which will have been mistakenly assigned session number 1) has
  been manually renamed or all of its outputs have been deleted.
