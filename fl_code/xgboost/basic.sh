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

if [ ! -d ./log/all/ ]; then
  mkdir -p ./log/all/
fi
rm -f ./log/all/*
[ -f ./$file_name ] && rm -f $file_name

while true; do
  ss -tlnp | grep $port > /dev/null
  if [ $? -eq 0 ]; then
    let port++
  else
    break
  fi
done

nohup python xgboost_server.py $port $client_num >./log/all/server.log 2>&1 &
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

for ((i = 0; i <= $client_num - 1; i++)); do
  nohup python xgboost_client.py $i -1 >./log/all/client$i.log 2>&1 &
  if [ $? -eq 0 ]; then
    echo "submit xgboost_client $i"
  else
    echo "submit xgboost_client $i error" && exit
  fi
done

echo "all task submitted"
