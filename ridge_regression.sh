#!/bin/bash
#
# Assumption: Must be run under the project directory

function run_hyperparam_search() {
  local model_type="$1"
  local dataset="$2"
  local num_sample=$3
  local num_epoch=$4
  local eigenvalue_file="$5"
  local l2_coeff=$6

  echo "========== ${model_type} =========="
  echo "dataset = ${dataset}"
  echo "num_sample = ${num_sample}"

  local log_dir="log/${model_type}"
  local conf_dir="conf/${model_type}"

  local num_iter=$(( num_epoch * num_sample ))

  # Creates basic dirs
  mkdir -p ${log_dir}
  mkdir -p ${conf_dir}
  mkdir -p tmp/
  local py_main=${model_type}_main.py

  # Prepares eigenvalues files
  if [ ${model_type} == "ridge_regression" ]; then
    python3 python/scripts/get_ridge_regression_hess.py \
      --input_file ${dataset} \
      --output_file ${eigenvalue_file} \
      --alpha ${l2_coeff} \
      > ${log_dir}/get_ridge_regression_hess.log \
      2> ${log_dir}/get_ridge_regression_hess.err
  fi

  # Searches for lr_schedule: constant
  echo "$(date): ===== search constant:"
  for init_lr in 0.1 0.06 0.03 0.02 0.01 0.006 0.003 0.002 0.001 0.0006 0.0003 0.0002 0.0001; do
    echo "$(date):  init_lr: ${init_lr}..."

    # Skips if have experiment records before
    local prefix="constant_init-lr-${init_lr}_num-epoch-${num_epoch}"
    if [ -f ${log_dir}/${prefix}.log ]; then
      echo "$(date):      have records before, skipped"
      continue
    fi

    python3 python/${py_main} \
      --pseudo_random \
      --framework numpy \
      --train_data ${dataset} \
      --logging_conf_file conf/common.log_conf \
      --lr_schedule_conf_file conf/lr_schedule/constant.conf \
      --alpha ${l2_coeff} \
      --num_epoch ${num_epoch} \
      --init_lr ${init_lr} \
      > ${log_dir}/${prefix}.log \
      2> ${log_dir}/${prefix}.err
  done


  echo "$(date): ===== search cosine decay:"
  for init_lr in 0.1 0.06 0.03 0.02 0.01 0.006 0.003 0.002 0.001 0.0006 0.0003 0.0002 0.0001; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in 0.1 0.01 0.001 0.0001 0.00001 0; do
      echo "$(date):    min_lr: ${min_lr}"
      for t_0 in ${num_iter}; do
        echo "$(date):      t_0 : ${t_0}"
        for t_mul in 1.0; do
          echo "$(date):        t_mul : ${t_mul}"

          # Skips if have experiment records before
          local prefix="cosine_init-lr-${init_lr}_epoch-${num_epoch}"
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
t_0 = ${t_0}
t_mul = ${t_mul}
min_lr = ${min_lr}
EOF

          python3 python/${py_main} \
            --pseudo_random \
            --framework numpy \
            --train_data ${dataset} \
            --logging_conf_file conf/common.log_conf \
            --lr_schedule_conf_file ${conf_path}\
            --alpha ${l2_coeff} \
            --num_epoch ${num_epoch} \
            --init_lr ${init_lr} \
            > "${log_dir}/${prefix}.log" \
            2> "${log_dir}/${prefix}.err"
        done
      done
    done
  done

  echo "$(date): ===== search eigencurve with same kappa:"
  for init_lr in 0.1 0.06 0.03 0.02 0.01 0.006 0.003 0.002 0.001 0.0006 0.0003 0.0002 0.0001; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in "" 0.1 0.01 0.001 0.0001 0.00001 0; do
      echo "$(date):    min_lr: ${min_lr}"
      for beta in 1.000005 2.0; do

        # Skips if have experiment records before
        local prefix="lambda-dependent_init-lr-${init_lr}_epoch-${num_epoch}"
        prefix+="_min-lr-${min_lr}_beta-${beta}_mu-same"
        if [ -f ${log_dir}/${prefix}.log ]; then
          echo "$(date):          have records before, skipped"
          continue
        fi

        # Prepares conf file
        local conf_path="${conf_dir}/${prefix}.conf"

        python3 python/scripts/preprocess_eigenval.py \
          --input_file ${eigenvalue_file} \
          --abs \
          --output_file ${eigenvalue_file}.abs.scale.txt \
          > /dev/null

        python3 python/scripts/gen_eigen_dependent_schedule_conf.py \
          --beta ${beta} \
          --input_eigenval_file ${eigenvalue_file}.abs.scale.txt \
          --output_file ${conf_path} \
          --num_iter ${num_iter} \
          --min_lr ${min_lr} \
          > /dev/null

        python3 python/${py_main} \
          --pseudo_random \
          --framework numpy \
          --train_data ${dataset} \
          --logging_conf_file conf/common.log_conf \
          --lr_schedule_conf_file ${conf_path}\
          --alpha ${l2_coeff} \
          --num_epoch ${num_epoch} \
          --init_lr ${init_lr} \
          > "${log_dir}/${prefix}.log" \
          2> "${log_dir}/${prefix}.err"
      done
    done
  done

  # Searches for lr_schedule: inverse_time_decay
  echo "$(date): ===== search inverse time decay:"
  for init_lr in 0.1 0.06 0.03 0.02 0.01 0.006 0.003 0.002 0.001 0.0006 0.0003 0.0002 0.0001; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in 0.1 0.01 0.001 0.0001 0.00001; do
      local lambda=$(python3 -c "print((${init_lr} / ${min_lr} - 1) / (${init_lr} * (${num_iter} - 1)))")
      echo "$(date):    min_lr: ${min_lr} (lambda: ${lambda})"

      # Skips if have experiment records before
      local prefix="inv_init-lr-${init_lr}_epoch-${num_epoch}_lambda-${lambda}"
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
lambda = ${lambda}
EOF

      python3 python/${py_main} \
        --pseudo_random \
        --framework numpy \
        --train_data ${dataset} \
        --logging_conf_file conf/common.log_conf \
        --lr_schedule_conf_file ${conf_path}\
        --alpha ${l2_coeff} \
        --num_epoch ${num_epoch} \
        --init_lr ${init_lr} \
        > "${log_dir}/${prefix}.log" \
        2> "${log_dir}/${prefix}.err"
    done
  done

  echo "$(date): ===== search exponential decay:"
  for init_lr in 0.1 0.06 0.03 0.02 0.01 0.006 0.003 0.002 0.001 0.0006 0.0003 0.0002 0.0001; do
    echo "$(date):  init_lr: ${init_lr}"
    for min_lr in 0.1 0.01 0.001 0.0001 0.00001; do
      local decay_rate=$(python3 -c "import math; print(math.e ** ((math.log(${min_lr}) - math.log(${init_lr})) / float(${num_iter} - 1)))")
      echo "$(date):    min_lr: ${min_lr} (decay_rate: ${decay_rate})"

      # Skips if have experiment records before
      local prefix="exp_init-lr-${init_lr}_epoch-${num_epoch}"
      prefix+="_decay-rate-${decay_rate}"
      if [ -f ${log_dir}/${prefix}.log ]; then
        echo "$(date):      have records before, skipped"
        continue
      fi

      # Prepares conf file
      local conf_path="${conf_dir}/${prefix}.conf"
      cat << EOF > ${conf_path}
[general]
type = exponential_decay

[hyperparams]
decay_step = 1
decay_rate = ${decay_rate}
EOF

      python3 python/${py_main} \
        --pseudo_random \
        --framework numpy \
        --train_data ${dataset} \
        --logging_conf_file conf/common.log_conf \
        --lr_schedule_conf_file ${conf_path}\
        --alpha ${l2_coeff} \
        --num_epoch ${num_epoch} \
        --init_lr ${init_lr} \
        > "${log_dir}/${prefix}.log" \
        2> "${log_dir}/${prefix}.err"
    done
  done
}

function main() {
  export PYTHONPATH="python"

  local model_type=""
  local dataset=""
  local num_sample=0
  local lipschitz_const=0
  local num_epoch=0
  local l2_coeff=0
  local eigenvalue_file=""

  model_type="ridge_regression"
  dataset="data/a4a.txt"
  num_sample=4781
  num_epoch=1
  l2_coeff=0.001
  eigenvalue_file="tmp/sorted_eigenvalue_ridge-regression_a4a.txt"
  run_hyperparam_search \
    ${model_type} ${dataset} ${num_sample} ${num_epoch} ${eigenvalue_file} ${l2_coeff}

  model_type="ridge_regression"
  dataset="data/a4a.txt"
  num_sample=4781
  num_epoch=5
  l2_coeff=0.001
  eigenvalue_file="tmp/sorted_eigenvalue_ridge-regression_a4a.txt"
  run_hyperparam_search \
    ${model_type} ${dataset} ${num_sample} ${num_epoch} ${eigenvalue_file} ${l2_coeff}

  model_type="ridge_regression"
  dataset="data/a4a.txt"
  num_sample=4781
  num_epoch=25
  l2_coeff=0.001
  eigenvalue_file="tmp/sorted_eigenvalue_ridge-regression_a4a.txt"
  run_hyperparam_search \
    ${model_type} ${dataset} ${num_sample} ${num_epoch} ${eigenvalue_file} ${l2_coeff}

  model_type="ridge_regression"
  dataset="data/a4a.txt"
  num_sample=4781
  num_epoch=250
  l2_coeff=0.001
  eigenvalue_file="tmp/sorted_eigenvalue_ridge-regression_abalone-scale.txt"
  run_hyperparam_search \
    ${model_type} ${dataset} ${num_sample} ${num_epoch} ${eigenvalue_file} ${l2_coeff}
}

main "$@"
