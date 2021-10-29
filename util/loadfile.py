import argparse
import os
import yaml
from types import SimpleNamespace as SN


class loadconfig:
    def __init__(self, args):
        config_file = os.path.join(os.getcwd(), args.config_dir)
        file = open(config_file, 'r')
        self._dit = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

    def getpara(self, key):
        pdict = {}
        for k, v in self._dit[key].items():
            pdict[k] = v
        args = SN(**pdict)
        return args

    def save_config(self, file_name, save_path):
        file_path = os.path.join(str(save_path), str(file_name) + '.yaml')
        file = open(file_path, mode='w', encoding='utf-8')
        yaml.dump(self._dit, file)
        file.close()
