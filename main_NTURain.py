#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-30 15:37:08

import commentjson as json
from trainers_singlegpu import trainer
from utils import str2none

def main():
    # set parameters
    with open('options_derain.json', 'r') as f:
        args = json.load(f)
    args['resume'] = str2none(args['resume'])

    for key, value in args.items():
        print('{:<25s}: {:s}'.format(key, str(value)))

    # intialize the trainer
    trainer_ntu = trainer(args)

    # Begin training
    trainer_ntu.train()

if __name__ == '__main__':
    main()

