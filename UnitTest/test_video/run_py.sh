#!/bin/sh
#!/bin/sh
ps -fe| grep make_video | grep -v grep
if [ $? -ne 0 ]
then
echo "start process mode 0"
(python3 make_video.py 0)&
echo "start process mode 1"
(python3 make_video.py 1)&
#echo "start process mode 2"
#(python3 make_video.py 2)&
else
echo "server had started....."
fi
