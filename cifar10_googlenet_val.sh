#!/bin/bash
#
# Assumption: Must be run under the project directory

function run_hyperparam_search() {
  local model_type=$1
  local dataset=$2
  local num_sample=$3
  local num_epoch=$4

  local model=${model_type}
  local batch_size=128
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

  local log_dir="log/cifar10/${model_type}_val"
  local conf_dir="conf/cifar10/${model_type}_val"

  # Creates basic dirs
  mkdir -p ${log_dir}
  mkdir -p ${conf_dir}
  local py_main=othernet_main.py

  echo "$(date): ===== search lambda-dependent with same kappa:"
  for init_lr in 1.0 0.6 0.3 0.2 0.1; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in "" 0 0.01 0.001 0.0001; do
      echo "$(date):    min_lr: ${min_lr}"

      # Skips if have experiment records before
      local prefix="eigencurve_lambda-dependent_init-lr-${init_lr}_epoch-${num_epoch}"
      prefix+="_min-lr-${min_lr}_mu-same"
      if [ -f ${log_dir}/${prefix}.log ]; then
        echo "$(date):          have records before, skipped"
        continue
      fi

      # Prepares conf file
      local conf_path="${conf_dir}/${prefix}.conf"

      python3 python/scripts/preprocess_eigenval.py \
        --input_file tmp/sorted_eigenvalue_model-${model_type}_iter-3000.txt \
        --abs \
        --output_file tmp/sorted_eigenvalue_model-${model_type}_iter-3000.abs.scale.txt \
        > /dev/null

      python3 python/scripts/gen_eigen_dependent_schedule_conf.py \
        --input_eigenval_file tmp/sorted_eigenvalue_model-${model_type}_iter-3000.abs.scale.txt \
        --output_file ${conf_path} \
        --num_iter ${num_iter} \
        --min_lr ${min_lr} \
        > /dev/null

      python3 python/${py_main} \
        --val \
        --model ${model} \
        --weight_decay  0.0005 \
        --batch_size ${batch_size} \
        --dataset ${dataset} \
        --logging_conf_file conf/common.log_conf \
        --lr_schedule_conf_file ${conf_path} \
        --num_epoch ${num_epoch} \
        --init_lr ${init_lr} \
        > ${log_dir}/${prefix}.log \
        2> ${log_dir}/${prefix}.err
    done
  done

  echo "$(date): ===== search cosine decay:"
  for init_lr in 1.0 0.6 0.3 0.2 0.1; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in 0.01 0.001 0.0001 0; do
      echo "$(date):    min_lr: ${min_lr}"
      for t_0 in ${num_iter_per_restart}; do
        echo "$(date):      t_0 : ${t_0}"
        for t_mul in 1.0; do
          echo "$(date):        t_mul : ${t_mul}"

          # Skips if have experiment records before
          local prefix="cosine-decay_cosine_activate-since-${activation_portion}_restart-${num_restart}_init-lr-${init_lr}_epoch-${num_epoch}"
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
t_0 = ${t_0}
t_mul = ${t_mul}
min_lr = ${min_lr}
EOF

          python3 python/${py_main} \
            --val \
            --model ${model} \
            --weight_decay 0.0005 \
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

  echo "$(date): ===== search inverse time decay:"
  for init_lr in 1.0 0.6 0.3 0.2 0.1; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in 0.01 0.001 0.0001 0.00001 0.000001; do
      local lambda=$(python3 -c "print((${init_lr} / ${min_lr} - 1) / (${init_lr} * (${num_iter_per_restart} - 1)))")
      echo "$(date):    min_lr: ${min_lr} (lambda: ${lambda})"

      # Skips if have experiment records before
      local prefix="inverse-time-decay_inv_activate-since-${activation_portion}_restart-${num_restart}_init-lr-${init_lr}_epoch-${num_epoch}_min-lr-${min_lr}_lambda-${lambda}"
      if [ -f ${log_dir}/${prefix}.log ]; then
        echo "$(date):      have records before, skipped"
        continue
      fi

      # Prepares conf file
      local conf_path="${conf_dir}/inversed-${lambda}.conf"
      cat << EOF > ${conf_path}
[general]
type = inverse_time_decay

[hyperparams]
activation_point = ${activation_point}
restarting_points = ${restarting_points}
lambda = ${lambda}
EOF

      python3 python/${py_main} \
        --val \
        --model ${model} \
        --weight_decay 0.0005 \
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

  echo "$(date): ===== common piecewise decay (used with momentum, but here no momentum used):"
  for init_lr in 1.0 0.6 0.3 0.2 0.1; do
    # Skips if have experiment records before
    local prefix="step-decay_resnet_piecewise_init-lr-${init_lr}_epoch-${num_epoch}_0.33-epoch-0.1_0.66-epoch-0.01_run-${run}"
    if [ -f ${log_dir}/${prefix}.log ]; then
      echo "$(date):          have records before, skipped"
      continue
    fi

    # Prepares conf file
    local conf_path="${conf_dir}/${prefix}.conf"
    cat << EOF > ${conf_path}
[general]
type = piecewise_constant

[hyperparams]
starting_points = $(python3 -c "print(${num_iter_per_restart} // 3)"), $(python3 -c "print(${num_iter_per_restart} // 3 * 2)")
factors = 0.1, 0.01
EOF
    python3 python/${py_main} \
      --val \
      --model ${model} \
      --weight_decay 0.0005 \
      --batch_size ${batch_size} \
      --dataset ${dataset} \
      --logging_conf_file conf/common.log_conf \
      --lr_schedule_conf_file ${conf_path}\
      --num_epoch ${num_epoch} \
      --init_lr ${init_lr} \
      > ${log_dir}/${prefix}.log \
      2> ${log_dir}/${prefix}.err
  done
}

function main() {
  export PYTHONPATH="python"
  local model_type="googlenet"
  local dataset="cifar10"
  local num_sample=50000
  local num_epoch=10
  run_hyperparam_search ${model_type} ${dataset} ${num_sample} ${num_epoch}
}

main "$@"
