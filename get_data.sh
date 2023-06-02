# Download the data, place it in the data folder, and run this script

if [ ! -f data/WebData_CROHME23.zip ]; then
    echo "File not found!"
    echo "Please download the dataset from https://uncloud.univ-nantes.fr/index.php/s/R9tWZSG3XeQbEeC/download"
    echo "and place it in the data folder"
    exit 1
fi

# If CROHME dataset folder does not exist, create it
mkdir -p data/CROHME/train/IMG
mkdir -p  data/CROHME/train/INKML
mkdir -p  data/CROHME/train/IMG_RENDERED
mkdir -p  data/CROHME/train/IMG_LABELS
mkdir -p  data/CROHME/train/SYNTHETIC
mkdir -p data/CROHME/train/SYNTHETIC_IMG_RENDERED
mkdir -p data/CROHME/val/IMG
mkdir -p  data/CROHME/val/IMG_RENDERED
mkdir -p  data/CROHME/val/INKML
mkdir -p  data/CROHME/val/SYNTHETIC
mkdir -p data/CROHME/val/IMG_LABELS

# Unzip the dataset
unzip data/WebData_CROHME23.zip -d data/CROHME/temp
mv data/CROHME/temp/WebData_CROHME23/* data/CROHME/temp

# Extract the contents of datasets within the Zip files. NOTE: THIS MAY NEED TO BE UPDATED
echo "Extracting the contents of the datasets within the Zip files..."
echo "Please update get_data.sh in the root directory if dataset structure changes. Please check the source if this fails."
unzip -q data/CROHME/temp/WebData_CROHME23_new_v2.3.zip -d data/CROHME/temp
unzip -q data/CROHME/temp/WebData_CROHME23_v1.1.zip -d data/CROHME/temp

# Copy the image files to the IMG folder
cp data/CROHME/temp/WebData_CROHME23/OffHME-dataset1/img/* data/CROHME/train/IMG/

# Copy the labels to the IMG_LABELS folder
cp data/CROHME/temp/WebData_CROHME23/OffHME-dataset1/label/* data/CROHME/train/IMG_LABELS/

# Copy the inkml files to the INKML folder
## Train Data
cp data/CROHME/temp/WebData_CROHME23/train/INKML/CROHME2019_*/* data/CROHME/train/INKML/
cp data/CROHME/temp/WebData_CROHME23_new_v2.3/new_train/INKML/* data/CROHME/train/INKML/

## Validation Data
cp data/CROHME/temp/WebData_CROHME23/val/INKML/CROHME2016_test/* data/CROHME/val/INKML/
cp data/CROHME/temp/WebData_CROHME23_new_v2.3/new_val/INKML/* data/CROHME/val/INKML/

# Get input from the user if they want to extract synthetic data
echo "Do you want to extract the synthetic data? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
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
    else
        echo "Skipping synthetic data extraction..."
fi

# Cleaning up the temp folder
rm -rf data/CROHME/temp