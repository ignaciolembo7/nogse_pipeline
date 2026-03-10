export SUBJECTS_DIR="$PWD/Data-signal-extracted/DATA_PROCESSED"
mkdir -p "$SUBJECTS_DIR"

recon-all -sd "$SUBJECTS_DIR" -s sub-20230619_BRAIN-3 -i Data-NIFTI-BRAINS-denoised_topup/20230619_BRAIN-3/20230619_BRAIN-3_t1_MEmprage_sag_p2_1mm-iso_freesurfer_QC-ROUTINE_20230619152408_3.nii.gz -all || echo "FALLÓ sub-20230619_BRAIN-3"

recon-all -sd "$SUBJECTS_DIR" -s sub-20230623_BRAIN-4 -i Data-NIFTI-BRAINS-denoised_topup/20230623_BRAIN-4/20230623_BRAIN-4_t1_MEmprage_sag_p2_1mm-iso_freesurfer_Anonymous_20230623084632_3.nii.gz -all || echo "FALLÓ sub-20230623_BRAIN-4"

recon-all -sd "$SUBJECTS_DIR" -s sub-20230622_BRAIN-1 -i Data-NIFTI-BRAINS-denoised_topup/20220622_BRAIN/t1_MEmprage_sag_p2_1mm-iso_freesurfer_3.nii.gz -all || echo "FALLÓ sub-20230622_BRAIN-1"

recon-all -sd "$SUBJECTS_DIR" -s sub-20230623_LUDG-2 -i Data-NIFTI-BRAINS-denoised_topup/20230623_LUDG-2/20230623_LUDG-2_t1_MEmprage_sag_p2_1mm-iso_freesurfer_Anonymous_20230623105657_4.nii.gz -all || echo "FALLÓ sub-20230623_LUDG-2"

recon-all -sd "$SUBJECTS_DIR" -s sub-20230629_MBBL-2 -i Data-NIFTI-BRAINS-denoised_topup/20230629_MBBL-2/20230629_MBBL-2_t1_MEmprage_sag_p2_1mm-iso_freesurfer_QC-ROUTINE_20230629150947_3.nii.gz -all || echo "FALLÓ sub-20230629_MBBL-2"




