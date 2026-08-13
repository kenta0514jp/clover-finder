#!/bin/sh
# Fetch the training set (55 MB, CC BY 4.0, by deton). Not committed here.
set -e
mkdir -p data && cd data
[ -f detect4clover.7z ] || curl -L -o detect4clover.7z \
  https://github.com/deton/detect4clover/releases/download/v1.0.0/detect4clover20200621.7z
python3 -c "import py7zr,sys; py7zr.SevenZipFile('detect4clover.7z','r').extractall('.')"
echo "extracted: $(ls img | wc -l) images, $(ls xml | wc -l) annotations"
