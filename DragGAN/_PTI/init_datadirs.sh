# Author: Batakala Ashok
# This script creates the directory structure for the experiment
# Usage: init_datadirs.sh <experiment_name> <path>
#==============================================================================


# if "--help" is given, print help
# if [ "$1" == "--help" ]
# then
#     echo "Usage: init_datadirs.sh <experiment_name> <path>"
#     exit 0
# fi


# if arg2 is not given, use current directory
if [ -z "$2" ]
then
    echo "No path given, using current directory"
    path="."
else
    path=$2
fi

cd $path

# check if experiment directory already exists
if [ -d "$1" ]
then
    echo "Experiment directory already exists"
    exit 1
fi
mkdir -p $1
cd $1
mkdir -p raw cropped aligned after_drag latents tuned_SG  edit_pasted videos landmarks before_drag image_show
touch log.txt


