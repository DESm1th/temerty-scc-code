#Set up:

  1. Clone this repo
  2. Clone the [Datman repo](https://github.com/TIGRLab/datman)
  3. Create private modules for both (see 'Private Modules' section below)
  4. Ensure the some version of the following modules are loaded before running
     the pipeline (in addition to use.own and the datman + qc-pipeline modules):
     .* Python 2.X (PYTHON/2.7.8-anaconda-2.1.0 is the one I used on the scc)
     .* AFNI
     .* FSL
     .* MINC
  5. Ensure the following packages are installed. "pip install --user <package_name>"
     will install each locally and automatically for the version of python you have
     loaded.
     .* docopt
     .* pydicom
  6. For each project to be managed, create a directory for logs and metadata.
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

  ### Datman module template

        #%Module
        proc ModulesHelp { } {
        	puts stderr "This module loads datman (https://github.com/TIGRLab/datman)"
        }

        # Load dependencies
        module load PYTHON/2.7.8-anaconda-2.1.0

        prepend-path PATH 	~/current/datman
        prepend-path PATH 	~/current/datman/bin
        prepend-path PATH	  ~/current/datman/datman

        prepend-path PYTHONPATH	~/current/datman

   ### qc-pipeline module template

        #%Module
        proc ModulesHelp { } {
        	puts stderr "This module loads the (temerty) versions of some datman scripts"
        }

        # Load dependencies
        module load PYTHON/2.7.8-anaconda-2.1.0

        prepend-path PATH	~/current/temerty-scc-code


# Config Templates
  ### project-settings.yml
  This file will be unreadable if there are any tabbed spaces. It's a good idea
  to use a text editor that automatically converts tabs to spaces (like atom) to
  edit it to avoid weird errors.

  The first value after 'ExportInfo' represents a tag that will be added to make
  it easy to locate members of that series. The pattern is the pattern that is
  in the SeriesDescription of the dicom headers. The site (CAMH in this case)
  is matched against the InstitutionName field of the header to determine which
  export info to use in the case of multiple scan sites.

    ###############
    # NEUR2MR Project Settings
    ###############

    STUDYNAME : NEUR2MR

    Sites:
      - CAMH :
          ## The data to expect from this project
          # Pattern: the regex pattern used to identify this scan from the dicom headers
          # Count: the number of scans of this type to expect in a session
          ExportInfo:
            - LOC:           { Pattern: '3PlaneLoc',              Count: 1}
            - T1:            { Pattern: 'T1',                     Count: 1}
            - RST:           { Pattern: 'Resting',                Count: 2}
            - GNG:           { Pattern: 'GoNoGo',                 Count: 3}
            - NBK:           { Pattern: 'Back',                   Count: 1}
            - DTI60-1000:    { Pattern: 'DTI-60plus5',            Count: 1}
            - T2:            { Pattern: 'T2',                     Count: 1}
            - FMAP-6.5:      { Pattern: 'TE65',                   Count: 1}
            - FMAP-8.5:      { Pattern: 'TE85',                   Count: 1}
            - ANI:           { Pattern: 'Fractional-Aniso',       Count: 1}

  ### exportinfo.csv
  This file will be phased out eventually. It's almost completely redundant
  with the information present in project-settings.yml's ExportInfo, but is
  current still needed as an input to convert_scans.py

    pattern     	    tag        export_nii  	export_nrrd export_mnc   	count
    3PlaneLoc	        LOC		        yes		    no          no              1
    T1		            T1		        yes		    no	        no              1
    Resting	          RST		        yes		    no	        no              2
    GoNoGo		        GNG		        yes		    no	        no              3
    NBack		          NBK		        yes		    no	        no              1
    DTI-60plus5	      DTI60-1000    yes		    no	        no              1
    T2		            T2		        yes		    no	        no              1
    TE65		          FMAP-6.5	    yes		    no	        no              1
    TE85		          FMAP-8.5	    yes		    no	        no              1
    Aniso		          ANI		        yes		    no	        no              1

  ### blacklist.csv
  Another input to convert_scans.py. Not super useful unless you intend to use
  the rest of the datman pipeline. The series column should contain the datman
  style name of a series that has been blacklist. Reason can be anything, as
  long as it doesn't contain spaces (this breaks the parsing)

      series									                                          reason
      STUDYNAME_SITENAME_SUBJECTID_TIMEPOINT_SESSION_SERIESDESCRIPTION  just-because
