source dataset_mito/config.sh

# Function for processing a single model
reorganize() {

  build_path=$1
  modelname="$(basename -- $3 .off)"
  output_path="$2/$modelname"
  echo $modelname

  points_file="$build_path/4_points/$modelname.npz"
  points_out_file="$output_path/points.npz"

  emitters_file="$build_path/4_emitters/$modelname.npz"
  emitters_out_file="$output_path/emitters.npz"

  pointcloud_file="$build_path/4_pointcloud/$modelname.npz"
  pointcloud_out_file="$output_path/pointcloud.npz"

  vox_file="$build_path/4_voxels/$modelname.binvox"
  # vox_file="$choy_vox_path/$modelname/model.binvox"
  vox_out_file="$output_path/model.binvox"

  img_dir="$build_path/5_multi_view_conf/N_${modelname}_*.tif"
  img_out_dir="$output_path/img_/"

  echo "Copying model $output_path"
  mkdir -p $output_path $img_out_dir

  rsync -a $img_dir $img_out_dir
  echo "Copied " $modelname

}

export -f reorganize

# Make output directories
mkdir -p $OUTPUT_PATH

# Create an all folder
OUTPUT_PATH_ALL=$OUTPUT_PATH/all
mkdir -p $OUTPUT_PATH_ALL

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  mkdir -p $OUTPUT_PATH_C


  ls $BUILD_PATH_C/0_in/*.off | parallel -P $NPROC --timeout $TIMEOUT \
    'reorganize' $BUILD_PATH_C $OUTPUT_PATH_C {} 

  echo "Creating split"
  python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.2

  # cp -r $OUTPUT_PATH_C/* $OUTPUT_PATH_ALL
  break
done

python create_split.py $OUTPUT_PATH_ALL --r_val 0.1 --r_test 0.2 --shuffle

