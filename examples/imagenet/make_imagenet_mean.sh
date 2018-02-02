#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/gauenk/Documents/data/ilsvrc/ILSVRC/lmdb/DET/ILSVRC2016_trainval1_lmdb
DATA=data/ilsvrc12
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/data.mdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
