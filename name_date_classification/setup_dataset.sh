#!/bin/bash

# Step 1: Clone the repository
module load git
git lfs install
echo "Cloning from Huggingface, Authentication required..."
git clone https://huggingface.co/datasets/ufdatastudio/mixtec-zouche-nuttall-british-museum

# Step 2: Rename the folder to "sign_images"
new_folder_name="sign_images"
mv mixtec-zouche-nuttall-british-museum "$new_folder_name"

# Step 3: Change into the new directory
cd "$new_folder_name" || exit

# Step 4: Delete the specified folders and files
rm -rf README.md figure-cutouts assets metadata.csv scene-cutouts .git .gitattributes

# Step 5: Move the contents of "name-date-cutouts" to the current directory
# and then delete the folder
if [ -d "name-date-cutouts" ]; then
  mv name-date-cutouts/* .
  rm -rf name-date-cutouts
fi

# Step 6: Delete the metadata.csv file (check if it exists)
rm -f metadata.csv

# Step 7: Define an array of keywords to use for folder creation and file categorization
keywords=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")

# Step 8: Create directories for each keyword inside train and test directories
for keyword in "${keywords[@]}"; do
  mkdir -p "train/$keyword" "test/$keyword"
done

# Set the percentage of files to go into the train set
train_percentage=70  # Replace with the desired percentage

# Step 9: Rename files ending in "-1.png" by removing "-1"
for file in *-1.png; do
  if [[ -e $file ]]; then
    new_name=$(echo "$file" | sed 's/-1\.png$/.png/')
    mv "$file" "$new_name"
    echo "Renamed $file to $new_name"
  fi
done

# Function to extract the class number from Format 1 files: `-<number><word>.png`
extract_class_format1() {
  filename=$1
  echo "$filename" | awk -F'-' '{print $NF}' | sed -E 's/[a-zA-Z]+\.png$//'
}

# Function to extract the class number from Format 2 files: `-year<number><word>.png`
extract_class_format2() {
  filename=$1
  echo "$filename" | sed -E 's/.*-year([0-9]+)[a-zA-Z]+\.png$/\1/'
}

# Function to randomly shuffle files
random_shuffle() {
  if command -v shuf >/dev/null 2>&1; then
    shuf -e "$@"
  else
    for file in "$@"; do
      echo "$file"
    done | awk 'BEGIN{srand()} {print rand(), $0}' | sort -k1,1n | cut -d" " -f2-
  fi
}

# Step 10: Process files for Format 1
for file in *.png; do
  class_number=$(extract_class_format1 "$file")
  if [[ $class_number =~ ^[0-9]+$ ]]; then
    train_dir="train/$class_number"
    test_dir="test/$class_number"
    
    mkdir -p "$train_dir" "$test_dir"
    
    # Randomly decide whether the file goes into train or test
    if ((RANDOM % 100 < train_percentage)); then
      mv "$file" "$train_dir/"
    else
      mv "$file" "$test_dir/"
    fi
  fi
done

# Step 11: Process remaining files for Format 2
for file in *.png; do
  class_number=$(extract_class_format2 "$file")
  if [[ $class_number =~ ^[0-9]+$ ]]; then
    train_dir="train/$class_number"
    test_dir="test/$class_number"
    
    mkdir -p "$train_dir" "$test_dir"
    
    # Randomly decide whether the file goes into train or test
    if ((RANDOM % 100 < train_percentage)); then
      mv "$file" "$train_dir/"
    else
      mv "$file" "$test_dir/"
    fi
  else
    echo "Skipping file: $file (no matching format)"
  fi
done

echo "Dataset setup and classification completed successfully!"

echo "Running Python Script to perform augmentation by random rotation"

# Step 12: Run the Python script to augment images with random rotations and color jitter

# Initialize conda
module load mamba
if command -v mamba &> /dev/null; then
    echo "Mamba is available. Initializing mamba..."
    mamba run --live-stream -n name-date python ../augment_images.py . 
else
    echo "Mamba is not available. Initializing conda..."
    conda run --live-stream -n name-date python ../augment_images.py . 
fi
