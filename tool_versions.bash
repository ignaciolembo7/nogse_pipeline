{
  echo "Python: $(python -V)"
  echo "FSL: $(flirt -version 2>/dev/null || echo 'unknown')"
  echo "ANTs: $(antsRegistration --version 2>/dev/null || echo 'unknown')"
  echo "MRtrix: $(mrconvert -version 2>/dev/null | head -n 1 || echo 'unknown')"
  echo "FreeSurfer: $(recon-all -version 2>/dev/null || echo 'unknown')"
} > tool_versions.txt
