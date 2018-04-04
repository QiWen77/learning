#!/usr/bin/env python
# coding: utf-8
from pymongo import DESCENDING
from data_new import MPWorkDb
import ruamel.yaml as yaml

OUT_YAML = '../export/paths.yaml'

workflowsDb = MPWorkDb().collection('fw')['workflows']
fireworksDb = MPWorkDb().collection('fw')['fireworks']
launchersDb = MPWorkDb().collection('fw')['launches']
workflows = workflowsDb.find({'state': 'COMPLETED'}, {'_id': 1, 'nodes': 1})


def get_fw_dict(fw_id):
    task_type = fireworksDb.find_one({
        'fw_id': fw_id
    }, {'_id': 0,
        'spec.task_type': 1})['spec']['task_type']
    launchers = launchersDb.find(
        {
            'fw_id': fw_id
        }, {'_id': 0,
            'launch_dir': 1},
        sort=[('_id', DESCENDING)])
    launch_dirs = [l['launch_dir'] for l in launchers]
    return {'fw_id': fw_id, 'task_type': task_type, 'launch_dirs': launch_dirs}


def get_start_fw_id_and_fws(fw_ids):
    start_fw = fireworksDb.find_one({
        'fw_id': min(fw_ids),
        'spec.snl.about.remarks': {
            '$in': ['I-42d', '1stRun']
        }
    }, {'_id': 0,
        'fw_id': 1})
    if start_fw:
        return min(fw_ids), [get_fw_dict(fw_id) for fw_id in fw_ids]
    else:
        return min(fw_ids), []


def get_paths(fw_ids):
    start_fw_id, fws = get_start_fw_id_and_fws(fw_ids)
    paths_dit = {}
    if fws:
        for fw in fws:
            if 'static' in fw['task_type']:
                paths_dit.update({'static': fw['launch_dirs']})
            if 'Uniform' in fw['task_type']:
                paths_dit.update({'uniform': fw['launch_dirs']})
            if 'band structure' in fw['task_type']:
                paths_dit.update({'band structure': fw['launch_dirs']})
    return {start_fw_id: paths_dit}


paths = {}
for wf in workflows:
    paths_wf = get_paths(wf["nodes"])
    if list(paths_wf.values())[0]:
        paths.update(paths_wf)
with open(OUT_YAML, 'w') as yaml_file:
    yaml.dump(paths, yaml_file, default_flow_style=False)
