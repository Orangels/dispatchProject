#!/bin/sh
ps -fe|grep app.py | grep -v grep
if [ $? -ne 0 ]
then
echo "start process rpc server"
(python3 rpcServer.py)&
sleep 20
echo "start process web server"
(python3 app.py)&

else
echo "server had started....."
fi