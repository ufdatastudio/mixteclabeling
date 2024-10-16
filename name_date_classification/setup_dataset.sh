#!/bin/bash

# Step 1: Clone the repository
echo "Cloning from Huggingface, Authentication required..."
git clone https://huggingface.co/datasets/ufdatastudio/mixtec-zouche-nuttall-british-museum


# Step 2: Rename the folder to "name_date_images"
new_folder_name="name_date_images"
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

# Step 7: Create 'train' and 'test' directories with 'date' and 'name' subdirectories
mkdir -p train/name_date train/year test/name_date test/year

# Set the percentage of files to go into the train set
train_percentage=75  # Replace with the desired percentage

# Step 8: Function to randomly shuffle files using available commands
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

# Step 9: Split files into 'train' and 'test' sets randomly
# Find all files with "year" in the filename and extension ".png"
year_files=(*year*.png)
num_year_files=${#year_files[@]}
train_year_count=$((num_year_files * train_percentage / 100))

# Randomly sort and move files with "year" to "train/date" and "test/date"
random_shuffle "${year_files[@]}" | {
  i=0
  while read -r file; do
    if [ $i -lt $train_year_count ]; then
      mv "$file" train/year/
    else
      mv "$file" test/year/
    fi
    ((i++))
  done
}

# Find all remaining .png files (those without "year" in the filename)
remaining_files=(*.png)
num_remaining_files=${#remaining_files[@]}
train_remaining_count=$((num_remaining_files * train_percentage / 100))

# Randomly sort and move remaining files to "train/name" and "test/name"
random_shuffle "${remaining_files[@]}" | {
  i=0
  while read -r file; do
    if [ $i -lt $train_remaining_count ]; then
      mv "$file" train/name_date/
    else
      mv "$file" test/name_date/
    fi
    ((i++))
  done
}

echo "Dataset setup and random splitting completed successfully!"