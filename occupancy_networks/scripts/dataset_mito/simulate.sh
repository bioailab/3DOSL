#!/usr/bin/env bash
# source dataset_mito/config.sh

# Make output directories
# trap "set +x; sleep 10; set -x" DEBUG
NPROC=0
BUILD_PATH=/data_mnt/data/EMPIAR-10791/data/mitochondria/mito.build/
mkdir -p $BUILD_PATH
CLASSES=1857

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  # input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c
  echo $build_path_c

          # $build_path_c/0_filled \
  mkdir -p   $build_path_c/5_2d_gt_conf
            $build_path_c/5_multi_conf \
            $build_path_c/5_epi_img \
            $build_path_c/5_epi_gt \    

  echo "Simulating images (In local sim env)"
  python3 simulate_img.py $build_path_c/5_emitters \
  --n_proc $NPROC  \
  --microscope_type confocal \
  --mic_conf dataset_mito/simulation/configs/conf2.py \
  --write_gt True \
  --img_folder $build_path_c/5_conf \
  --gt_folder $build_path_c/5_2d_gt_conf \

  echo "Simulating images (In local sim env)"
  python3 simulate_img.py $build_path_c/5_emitters \
  --n_proc $NPROC  \
  --microscope_type epi \
  --mic_conf dataset_mito/simulation/configs/conf2.py \
  --write_gt True
  --img_folder $build_path_c/5_epi_img \
  --gt_folder $build_path_c/5_epi_gt \

done
