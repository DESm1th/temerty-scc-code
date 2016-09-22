#!/bin/bash

# This script is intended to be used when one or more scans have
# stored the correct participant ID in the PatientID field
# of the dicom header instead of the PatientName field.

# The headers will be changed so both fields contain the intended ID
# and the folder containing the scan is renamed to the correct ID.

path=$1

# Remove / at end of path if present
path=${path%/}

for pt_folder in $path/*
do
  # Retrieve the current scanid from the folder name
  scanid=${pt_folder##*/}

  for dcm in `find $pt_folder -name "*.dcm"`
  do
    # Retrieve the intended scanid from a single dicom header
    ID=`dcm_header.py --field PatientID ${dcm}`

    if [ $scanid != $ID ]
    then
      # Get the path to the scan, in case it differs from pt_folder
      # It's assumed that scan structure is scan/series_folders/dicoms
      series_path="${dcm%/*.dcm}"
      scan_path="${series_path%/*}"

      # Fix the headers, to avoid the rest of the pipeline using the
      # bad value stored in patientName
      anonymize_headers.py --patientID ${scan_path}

      # Rename the folder that holds the scan to match the intended ID
      new_path="${path}/${ID}"
      mv $pt_folder $new_path
    fi
    break
  done
done
