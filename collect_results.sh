#!/bin/bash

model="resnet18"
for epoch in 1 5 30 90 270; do
  echo "===== #Epoch = ${epoch}"
  echo "Top-1_acc Top-5_acc Training_loss"
  for file in log/run_imagenet/${model}/cosine*_epoch-${epoch}_*.log; do
    echo "$(cat ${file} \
      | tail -3 \
      | head -1 \
      | sed 's/.*train loss: \([^,]*\),.*test accuracy: \(.*\), test top-5 accuracy: \(.*\)$/\2  \3  \1/') ${file}";
  done
done
