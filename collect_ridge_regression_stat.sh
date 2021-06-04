#!/bin/bash

function main() {
  if [ $1 == '-h' ]; then
    printf "Usage: ./$(basename $0) {num_of_epochs}\n" >&2
    printf "Example: ./$(basename $0) 25\n" >&2
    return 0
  fi

  local epoch=$1
  local opt=$(tail -1 log/ridge_regression/constant_init-lr-0.1_num-epoch-${epoch}.log \
    | awk '{ print $NF }')
  echo "optimal loss = ${opt}"
  echo "loss-opt    loss    file"
  for schedule in constant inv exp cosine lambda-dependent; do
    record=$(grep -rn "min_loss" $(ls log/ridge_regression/* | grep "${schedule}" | grep "epoch-${epoch}[_\.]") \
          | awk -v opt=${opt} '{ print $NF - opt" "$NF" "$0 }' \
          | sed 's/\:.*$//' \
          | sort -n \
          | head -1)
    echo "${record}"
  done
}

main "$@"
