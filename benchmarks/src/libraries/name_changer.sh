#!/bin/bash

# Find all mve.cpp files in subdirectories and rename them to mve.cpp
for file in */*/mve.cpp; do
  if [ -f "$file" ]; then
    new_file="${file%mve.cpp}mve.cpp"
    # mv "$file" "$new_file"
    echo "Renamed: $file -> $new_file"
  fi
done

echo "All files have been renamed."