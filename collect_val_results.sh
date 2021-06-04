#!/bin/bash

epoch=100

for model in resnet googlenet vgg16; do
  echo "========== ${model}"
  for schedule in eigencurve inverse-time-decay cosine-decay step-decay; do
    record=$(grep -rn "End of epoch $(( ${epoch} - 1))" $(ls log/cifar10/${model}_val/* | grep "${schedule}" | grep "epoch-${epoch}") \
        | awk '{ print -$NF" "$0 }' \
        | sed 's/^\([^\ ]*\) \([^:]*\):.*$/\1 \2/' \
        | sed 's/,//' \
        | sort -n \
        | head -1)
    file=$(echo "${record}" | awk '{print $2}')
    echo "${record} $(tail -3 ${file} | head -1 | sed 's/.*test accuracy: \(.*\)/\1/')"
  done
done
