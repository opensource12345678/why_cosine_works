#!/bin/bash
#
# Assumption: Must be run under the project directory

function run_hyperparam_search() {
  local model_type=$1
  local dataset=$2
  local num_sample=$3
  local num_epoch=$4

  local model=${model_type}
  local batch_size=256
  local activation_portion=0
  local num_restart=0

  local num_iter_per_epoch=$(python3 -c "print((${num_sample} + ${batch_size} - 1) // ${batch_size})")
  local num_iter=$(( num_epoch * num_iter_per_epoch ))

  local activation_point=$(python3 -c "print(int(${num_iter} * ${activation_portion}))")
  local num_iter=$(python3 -c "print(int(${num_iter} * (1 - ${activation_portion})))")

  local restarting_points=0
  if [ ${num_restart} -gt 0 ]; then
    restarting_points=$(python3 -c "print(','.join([str((i + 1) * int(${num_iter} // (${num_restart} + 1))) for i in range(${num_restart})]))")
  fi
  local num_iter_per_restart=$(python3 -c "print(${num_iter} // (${num_restart} + 1))")

  echo "activation_portion = ${activation_portion}"
  echo "activation_point = ${activation_point}"
  echo "num_restart = ${num_restart}"
  echo "restarting_points = ${restarting_points}"

  echo "========== ${model_type} =========="
  echo "dataset = ${dataset}"
  echo "num_sample = ${num_sample}"
  echo "num_epoch = ${num_epoch}"
  echo "batch_size = ${batch_size}"

  local log_dir="log/run_imagenet/${model_type}"
  local conf_dir="conf/run_imagenet/${model_type}"

  # Creates basic dirs
  mkdir -p ${log_dir}
  mkdir -p ${conf_dir}
  local py_main=othernet_main.py

  echo "$(date): ===== search cosine decay:"
  for init_lr in 1.0; do
    echo "$(date):  init_lr: ${init_lr}"
    for power in 1.0 0.5 2.0; do
      echo "$(date):    power: ${power}"
      for min_lr in 0; do
        echo "$(date):      min_lr: ${min_lr}"
        for t_0 in ${num_iter_per_restart}; do
          echo "$(date):        t_0 : ${t_0}"
          for t_mul in 1.0; do
            echo "$(date):          t_mul : ${t_mul}"

            # Skips if have experiment records before
            local prefix="cosine-power-${power}_activate-since-${activation_portion}_restart-${num_restart}_init-lr-${init_lr}_epoch-${num_epoch}"
            prefix+="_min-lr-${min_lr}_t-0-${t_0}_t-mul-${t_mul}"
            if [ -f ${log_dir}/${prefix}.log ]; then
              echo "$(date):          have records before, skipped"
              continue
            fi

            # Prepares conf file
            local conf_path="${conf_dir}/${prefix}.conf"
            cat << EOF > ${conf_path}
[general]
type = cosine_decay

[hyperparams]
activation_point = ${activation_point}
restarting_points = ${restarting_points}
power = ${power}
t_0 = ${t_0}
t_mul = ${t_mul}
min_lr = ${min_lr}
EOF

            python3 python/${py_main} \
              --model ${model} \
              --checkpoint_dir checkpoints \
              --checkpoint_name ${prefix} \
              --weight_decay 0.0001 \
              --batch_size ${batch_size} \
              --dataset ${dataset} \
              --logging_conf_file conf/common.log_conf \
              --lr_schedule_conf_file ${conf_path}\
              --num_epoch ${num_epoch} \
              --init_lr ${init_lr} \
              > ${log_dir}/${prefix}.log \
              2> ${log_dir}/${prefix}.err
          done
        done
      done
    done
  done
}

function main() {
  export PYTHONPATH="python"
  export CUDA_VISIBLE_DEVICES=0,1;

  local model_type="resnet18"
  local dataset="imagenet"
  local num_sample=1281167
  local num_epoch=1
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}

  local model_type="resnet18"
  local dataset="imagenet"
  local num_sample=1281167
  local num_epoch=5
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}

  local model_type="resnet18"
  local dataset="imagenet"
  local num_sample=1281167
  local num_epoch=30
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}

  local model_type="resnet18"
  local dataset="imagenet"
  local num_sample=1281167
  local num_epoch=90
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}

  local model_type="resnet18"
  local dataset="imagenet"
  local num_sample=1281167
  local num_epoch=270
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}
}

main "$@"
