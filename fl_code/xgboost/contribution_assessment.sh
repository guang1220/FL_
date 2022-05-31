#!/bin/bash
# Desc: ALL Client

sub_path=lib/python3.7/site-packages/flwr/common/parameter.py
root_path=`which python | sed 's@/bin/python@/@g'`
path=$root_path$sub_path
sed -r -i '29s/#+tensors/tensors/g' $path
sed -i '28s/tensors/#tensors/g' $path

file_name=server_IP
client_num=$1

port=8080
[ -f ./$file_name ] && rm -f $file_name

for ((i = 0; i < $client_num; i++)); do

  if [ ! -d ./log/lack_$i/ ]; then
    mkdir -p ./log/lack_$i/
  fi
  rm -f ./log/lack_$i/*

  while true; do
    ss -tlnp | grep $port > /dev/null
    if [ $? -eq 0 ]; then
      let port++
    else
      break
    fi
  done

  nohup python xgboost_server.py $port $(($client_num - 1)) >./log/lack_$i/server.log 2>&1 &
  if [ $? -eq 0 ]; then
    echo "submit xgboost_server $port"
  else
    echo "submit xgboost_server error" && exit
  fi

  count=0
  while [ ! -f ./$file_name ] && (($count < 10)); do
    sleep 6
    let count++
  done

  (($count >= 10)) && echo "no find $file_name" && exit

  for ((j = 0; j <= $(($client_num - 1)); j++)); do
    [ $j -eq $i ] && continue
    nohup python xgboost_client.py $j $i >./log/lack_$i/client$j.log 2>&1 &
    if [ $? -eq 0 ]; then
      echo "submit xgboost_client $j"
    else
      echo "submit xgboost_client $j error" && exit
    fi
  done

  echo "lack $i _client Contribution assessment task submitted"
  sleep 10

done

echo "all task submitted"
