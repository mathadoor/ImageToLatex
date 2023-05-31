# Download the data, place it in the data folder, and run this script

if [ ! -f data/WebData_CROHME23.zip ]; then
    echo "File not found!"
    echo "Please download the dataset from https://uncloud.univ-nantes.fr/index.php/s/R9tWZSG3XeQbEeC/download"
    echo "and place it in the data folder"
    exit 1
fi

# If CROHME dataset folder does not exist, create it
if [ ! -d data/CROHME ]; then
    mkdir -p data/CROHME/train/IMG
    mkdir data/CROHME/train/INKML
    mkdir data/CROHME/train/IMG_RENDERED
    mkdir data/CROHME/train/SYNTHETIC
    mkdir -p data/CROHME/val/IMG
    mkdir data/CROHME/val/IMG_RENDERED
    mkdir data/CROHME/val/INKML
    mkdir data/CROHME/val/SYNTHETIC
fi

# Unzip the dataset
unzip data/WebData_CROHME23.zip -d data/CROHME/temp
mv data/CROHME/temp/WebData_CROHME23/* data/CROHME/temp

# Extract the contents of datasets within the Zip files. NOTE: THIS MAY NEED TO BE UPDATED
echo "Extracting the contents of the datasets within the Zip files..."
echo "This may need to be updated if the dataset changes. Please check the source if this fails."
unzip -q data/CROHME/temp/WebData_CROHME23_new_v2.3.zip -d data/CROHME/temp
unzip -q data/CROHME/temp/WebData_CROHME23_v1.1.zip -d data/CROHME/temp
#
## Copy the image files to the IMG folder
cp data/CROHME/temp/WebData_CROHME23/OffHME-dataset1/img/* data/CROHME/train/IMG/

# Copy the inkml files to the INKML folder
cp data/CROHME/temp/WebData_CROHME23/train/INKML/CROHME2019_*/* data/CROHME/train/INKML/
cp data/CROHME/temp/WebData_CROHME23_new_v2.3/new_train/INKML/* data/CROHME/train/INKML/

# Copy the synthetic inkml files to the SYNTHETIC folder
echo "Copying the synthetic inkml files to the SYNTHETIC folder..."
echo "This will take a while..."
src="data/CROHME/temp/WebData_CROHME23_new_v2.3/Syntactic_data/INKML/"
dst="data/CROHME/train/SYNTHETIC/"

find "$src" -type f -exec sh -c '
  f="$1"
  base=$(basename "$f")
  parent=$(basename $(dirname "$f"))
  cp "$f" "$0/$parent-$base"
' "$dst" {} \;

rm -rf data/CROHME/temp