#!/usr/bin/env bash
# source dataset_mito/config.sh

# Make output directories
# trap "set +x; sleep 10; set -x" DEBUG
NPROC=0             # Set > 0  for parallel processing
BUILD_PATH=/data_mnt/EMPIAR-10791/data/mitochondria/no_hole.build/
mkdir -p $BUILD_PATH
CLASSES=1857

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  # input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c
  echo $build_path_c

          # $build_path_c/0_filled \
  mkdir -p   $build_path_c/6_conf2_img


  echo "Simulating confocal stacks (In local sim env)"
  python3 dataset_mito/simulate_stack.py $build_path_c/5_emitters \
  --n_proc $NPROC  \
  --microscope_type confocal \
  --mic_conf dataset_mito/simulation/configs/epi2_stack.py \
  --img_folder $build_path_c/6_conf2_img_stck \


done