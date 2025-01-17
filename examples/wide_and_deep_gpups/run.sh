# !/bin/bash
if [ ! -d "./log" ]; then
  mkdir ./log
  echo "Create log floder for store running log"
fi

export FLAGS_LAUNCH_BARRIER=0
export PADDLE_TRAINER_ID=0
export PADDLE_PSERVER_NUMS=1
export PADDLE_TRAINERS=1
export PADDLE_TRAINERS_NUM=${PADDLE_TRAINERS}
export POD_IP=127.0.0.1
export PSERVER_DEBUG=1
# export GLOG_v=1

# set free port if 29011 is occupied
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"
export PADDLE_PSERVER_PORT_ARRAY=(29011)

# set gpu numbers according to your device
export FLAGS_selected_gpus="0"

# run pserver
export TRAINING_ROLE=PSERVER
for((i=0;i<$PADDLE_PSERVER_NUMS;i++))
do
    cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
    echo "PADDLE WILL START PSERVER "$cur_port
    export PADDLE_PORT=${cur_port}
    python3.7 -u train.py &> ./log/pserver.$i.log &
done

# run trainer
export TRAINING_ROLE=TRAINER
for((i=0;i<$PADDLE_TRAINERS;i++))
do
    echo "PADDLE WILL START Trainer "$i
    export PADDLE_TRAINER_ID=$i
    python3.7 -u train.py &> ./log/worker.$i.log
done

echo "Training log stored in ./log/"