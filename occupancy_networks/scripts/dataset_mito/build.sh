#!/usr/bin/env bash
source dataset_mito/config.sh
# Make output directories
# trap "set +x; sleep 10; set -x" DEBUG

mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c


          # $build_path_c/0_in \
  mkdir -p $build_path_c/0_filled \
           $build_path_c/1_scaled \
           $build_path_c/1_transform \
           $build_path_c/2_depth \
           $build_path_c/2_watertight \
           $build_path_c/4_points \
           $build_path_c/4_pointcloud \
           $build_path_c/4_voxels \
           $build_path_c/4_watertight_scaled \
           $build_path_c/5_emitters \
           $build_path_c/5_epi_img \
           $build_path_c/5_epi_gt \

  # echo "Converting meshes to OFF"
  # echo $input_path_c/{}/model.obj
  # echo $build_path_c/0_in/{}.off
  # lsfilter $input_path_c $build_path_c/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
  #    meshlabserver -platform offscreen -i $input_path_c/{}/model.obj -o $build_path_c/0_in/{}.off;

# Modified to rendering with virtual display to run on server, offscreen rendering still requires some initializations from GUI 
# https://stackoverflow.com/questions/49799634/cant-run-meshlabserver-filters-on-a-headless-virtual-environment
  # lsfilter  $input_path_c $build_path_c/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
  # xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i $input_path_c/{}/model.obj -o $build_path_c/0_in/{}.off;
  # parallel_convert_meshes $input_path_c $build_path_c/0_in .off

  # parallel_fill_holes $build_path_c/0_in $build_path_c/0_filled
  # break


  echo "Scaling meshes"
  PYOPENGL_PLATFORM=egl python $MESHFUSION_PATH/1_scale.py \
    --n_proc $NPROC \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
  
  echo "Create depths maps"
  # echo $PWD
  xvfb-run -a -s "-screen 0 800x600x24" python $MESHFUSION_PATH/2_fusion.py \
    --mode=render --n_proc $NPROC \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth

  echo "Produce watertight meshes"
  xvfb-run -a -s "-screen 0 800x600x24" python $MESHFUSION_PATH/2_fusion.py \
    --mode=fuse --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/2_watertight \
    --t_dir $build_path_c/1_transform
    --overwrite \

  echo "Process watertight meshes"
  # PYOPENGL_PLATFORM=egl
  # xvfb-run -a -s "-screen 0 800x600x24" python sample_mesh.py $build_path_c/2_watertight \
  xvfb-run -a -s "-screen 0 800x600x24" python sample_mesh.py $build_path_c/2_watertight \
      --n_proc $NPROC --resize \
      --bbox_in_folder $build_path_c/0_in \
      --pointcloud_folder $build_path_c/4_pointcloud \
      --voxels_folder $build_path_c/4_voxels \
      --points_folder $build_path_c/4_points \
      --mesh_folder $build_path_c/4_watertight_scaled \
      --packbits --float16 \
      --overwrite

  # echo "Generate emitter locations"
  # xvfb-run -a -s "-screen 0 800x600x24" python sample_mesh.py $INPUT_PATH/$c \
  xvfb-run -a -s "-screen 0 800x600x24" python sample_mesh.py $build_path_c/0_in \
    --n_proc $NPROC  \
    --emitters_folder $build_path_c/5_emitters\
    --float16 \
    --overwrite \
    --resize
        # --bbox_in_folder $build_path_c/0_in \

  # echo "Simulating images (In local sim env)"
  # python3 simulate_img.py $build_path_c/5_emitters \
  # --n_proc $NPROC  \
  # --microscope_type confocal \
  # --img_folder $build_path_c/5_multi_view_conf \
  # --gt_folder $build_path_c/5_2d_gt_conf \

  # echo "Simulating images (In local sim env)"
  # python3 simulate_img.py $build_path_c/5_emitters \
  # --n_proc $NPROC  \
  # --microscope_type epi \
  # --write_gt True
  # --img_folder $build_path_c/5_epi_img \
  # --gt_folder $build_path_c/5_epi_gt \

  # echo "Simulating image stacks (In local sim env)"
  # python3 simulate_stak.py $build_path_c/5_emitters \
  # --n_proc $NPROC  \
  # --microscope_type epi \
  # --img_folder $build_path_c/5_multi_view_stack_epi \

  # break
done
