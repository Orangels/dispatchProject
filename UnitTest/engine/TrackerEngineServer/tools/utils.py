import time
import os
import platform
import traceback
import GPUtil as GPU
import psutil
import requests
import json
from urllib.parse import quote, unquote


def get_state_filepath(root_path, filename):
    year = time.strftime('%Y', time.localtime())
    month = time.strftime('%m', time.localtime())
    day = time.strftime('%d', time.localtime())
    # 日期地址
    date_path = year + '/' + month + '/' + day + '/'
    file_dir_path = root_path + date_path
    if not os.path.exists(file_dir_path):
        os.makedirs(file_dir_path)
    filePs = file_dir_path + filename
    return filePs, date_path


def get_gpu_load():
    if platform.machine().find("x86") == 0:
        try:
            GPUs = GPU.getGPUs()
            for gpu in GPUs:
                return int(gpu.load * 100)
        except:
            print(traceback.format_exc())
    elif platform.machine().find("aarch") == 0:
        try:
            gpuLoadFile = "/sys/devices/gpu.0/load"
            with open(gpuLoadFile, 'r') as gpuFile:
                fileData = gpuFile.read()
            return int(int(fileData) / 10)
        except:
            print(traceback.format_exc())
    else:
        pass
    return None


def cpu_temp():
    with open('/sys/class/hwmon/hwmon0/temp1_input', 'rt') as f:
        # 读取结果，并转换为整数
        temp = int(f.read()) / 1000
    return temp


def get_device_basic_info():
    cpu_usage_percent = psutil.cpu_percent(0)

    gpu_usage_percent = get_gpu_load()
    virtual_memory_total = size_humanize(psutil.virtual_memory().total)
    virtual_memory_usage = size_humanize(psutil.virtual_memory().used)

    disk_partitions = psutil.disk_partitions(all=True)  # 磁盘分区信息
    disk_total = 0
    disk_usage = 0
    for sdiskpart in disk_partitions:
        if sdiskpart.fstype:
            disk_total += psutil.disk_usage(sdiskpart.mountpoint).total
            disk_usage += psutil.disk_usage(sdiskpart.mountpoint).used

    return dict(cpu_percent=cpu_usage_percent, gpu_percent=gpu_usage_percent, cpu_temp=cpu_temp(),
                memory_total=virtual_memory_total, memory_used=virtual_memory_usage,
                disk_total=size_humanize(disk_total, "G"), disk_used=size_humanize(disk_usage, "G")
                )


def size_humanize(size, need_unit='M'):
    units = ['b', 'K', 'M', 'G', 'T', 'P',
             'E', 'Z', 'Y', 'B', 'N', 'D', 'C']

    try:
        size = int(size)
    except:
        print(traceback.format_exc())
        return False

    if size < 0:
        return False

    for unit in units:
        if size >= 1024 and unit != need_unit:
            size //= 1024
        else:
            size_h = '{}{}'.format(size, unit)
            return size_h

    size_h = '{}{}'.format(size, units[-1])
    return size_h


def get_date_str(timestamp):
    return time.strftime("%H:%M:%S", time.localtime(timestamp))


def postMethod(url, data):
    headers = {'Content-Type': 'application/json'}
    post_json = json.dumps(data)

    data = quote(post_json)
    r = requests.post(url, headers=headers, data=data)
    print(r.text)


if __name__ == "__main__":
    print(get_device_basic_info())