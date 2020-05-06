import yaml
import os

root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
config_path = os.path.join(root, 'config')


def get_config_from_dirs(file_path):
    config = dict()
    filelist = os.listdir(file_path)
    for f in filelist:
        sub_file_path = os.path.join(file_path, f)
        if os.path.isfile(sub_file_path):
            with open(sub_file_path, "r") as yaml_file:
                yaml_obj = yaml.load(yaml_file.read())
                config[(os.path.splitext(f))[0]] = yaml_obj
        elif os.path.isdir(sub_file_path):
            continue
            # 不向下递归
            # sub_config = get_filepath_from_dirs(sub_file_path)
            # config = dict(**config, **sub_config)
    return config


class yaml_config(object):
    def __init__(self, config_path=config_path):
        '''
        :param config: 文件夹, 读取文件夹下所有 config 文件, (config 以 cameraID 为文件名)
        '''
        assert os.path.exists(config_path), 'path is not exists'
        self._config = get_config_from_dirs(config_path)
        self._config_path = config_path

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_dict):
        '''
        :param config_dict: {
            camera_id:{config},
        }
        :return:
        '''
        assert isinstance(config_dict, dict), 'config_dict is not dict'
        for config_flag_key in config_dict.keys():
            config_path = self._config_path + '/' + config_flag_key + '.yaml'
            for key in config_dict[config_flag_key].keys():
                if config_flag_key in self._config.keys():
                    self._config[config_flag_key][key] = config_dict[config_flag_key][key]
                else:
                    self._config[config_flag_key] = {}
                    self._config[config_flag_key][key] = config_dict[config_flag_key][key]
            with open(config_path, "w") as yaml_file:
                yaml.dump(self._config[config_flag_key], yaml_file, default_flow_style=False, encoding='utf-8',
                          allow_unicode=True)


# class redis_config(object):

if __name__ == '__main__':
    y = yaml_config()
    print(y.config)
