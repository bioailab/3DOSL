ROOT=..
DATAROOT=/data_mnt/data/EMPIAR-10791         # SEt this to your data root
export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

# 
INPUT_PATH=$DATAROOT/data/objects_3_5     # Set this to the path where the mesh files in the fomat of .obj are are stored

FILL_HOLE_PATH=../../hole_fixer/build
BUILD_PATH=$DATAROOT/data/mito.build
OUTPUT_PATH=$DATAROOT/data/mito.install


NPROC=50
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
1857
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

#  echo "other_folder $other_folder, ext : $ext"
 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}

parallel_convert_meshes() {
  folder=$1
  other_folder=$2
  # ext=$3
  # echo "folder ${folder}, other_folder ${other_folder}, ext : ${ext}"
  for f in $folder/*; do
    filename=$(basename $f)
    basename="${filename%.*}"
    echo "filename $filename"
    xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i $input_path_c/${filename} -o $build_path_c/0_in/${basename}.off &
    # break
  done
}

parallel_fill_holes() {
  folder=$1
  other_folder=$2
  # ext=$3
  echo "folder ${folder}, other_folder ${other_folder}, ext : ${ext}"
  for f in $folder/*; do
    filename=$(basename $f)
    basename="${filename%.*}"
    echo "filename $filename"
    ./${FILL_HOLE_PATH}/hole_fixer -in $folder/${filename} -out $other_folder/${basename}.off -outfaces 8000 -upsample 2 &
    # break
  done
}


