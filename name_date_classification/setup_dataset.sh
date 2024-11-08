#!/bin/bash


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
keywords=(jaguar movement eagle flint flower wind rain dog rabbit reed grass crocodile serpent monkey deer vulture house death water lizard)

# Step 8: Create directories for each keyword inside train and test directories
for keyword in "${keywords[@]}"; do
  mkdir -p "train/$keyword" "test/$keyword"
done

# Set the percentage of files to go into the train set
train_percentage=70  # Replace with the desired percentage

# Function to randomly shuffle files
random_shuffle() {
  if command -v shuf >/dev/null 2>&1; then
    # If shuf is available, use it
    shuf -e "$@"
  else
    # Fallback method for macOS using awk and sort
    for file in "$@"; do
      echo "$file"
    done | awk 'BEGIN{srand()} {print rand(), $0}' | sort -k1,1n | cut -d" " -f2-
  fi
}

# Step 9: Categorize files based on keywords and split into train and test
for keyword in "${keywords[@]}"; do
  # Find files containing the keyword
  files=(*"$keyword"*.png)
  num_files=${#files[@]}
  
  if [ $num_files -eq 0 ]; then
    continue  # Skip if no files match the keyword
  fi
  
  train_count=$((num_files * train_percentage / 100))
  
  # Randomly shuffle and split files into train and test sets
  random_shuffle "${files[@]}" | {
    i=0
    while read -r file; do
      if [ $i -lt $train_count ]; then
        mv "$file" "train/$keyword/"
      else
        mv "$file" "test/$keyword/"
      fi
      ((i++))
    done
  }
done

echo "Dataset setup and random splitting completed successfully!"

echo "Running Python Script to perform augmentation by random rotation"

# Step 10: Run the Python script to augment images with random rotations and color jitter

# Initialize conda
if command -v mamba &> /dev/null; then
    # Mamba is available, so we initialize it
    echo "Mamba is available. Initializing mamba..."
    mamba run --live-stream -n name-date python augment_images.py . 
else
    # Fallback to conda if mamba is not available
    echo "Mamba is not available. Initializing conda..."
    conda run --live-stream -n name-date python ../augment_images.py . 
fi