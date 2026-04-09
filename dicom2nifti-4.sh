#!/bin/bash

# Step 1/30: This script converts DDE-CTI human experiments from 2021-04-14 to NIfTI
# Manu Raghavan, 04/03/2025
# Input: DICOM files from the rawdata folder
# Output: Converted NIfTI files that go to the sourcedata folder.

cd ../sourcedata

mypath="/mnt/storage/tier2/MUMI-EXT-001/mumi-data/USERS/Manu_Raghavan/CTI-Manu/rawdata" # Update[Manu]: changed this path to my CTI-Manu/rawdata folder

for fol in $mypath; do

  echo
  echo "Your current folder is: " $fol
  echo

  folname="$(basename $fol)"

  sub="sub-09"

  

  echo "Folder name is: " $folname
  echo
  echo "Current subject is: " $sub
  echo

  mkdir $sub

  echo "Here follow the detected series:"
  echo

  for ser in ${fol}/*; do

    echo
    echo $ser
    echo

    sername="$(basename $ser)"

    mkdir ./${sub}/${sername}

    dcm2niix -z y -o ./${sub}/${sername} $ser

  done

  echo "Dicom to NIfTI conversion done for " $folname

  echo $folname >> ./${sub}/${sub}_orig-MR-code.txt

done
