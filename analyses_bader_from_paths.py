import json
import logging
import os
import warnings

import ruamel.yaml as yaml
from pymatgen.command_line.bader_caller import bader_analysis_from_path

warnings.filterwarnings(
    'ignore', message='^Multiple files detected', module='pymatgen')
logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_JSON = 'out.json'

with open('paths.yaml') as yamlfile:
    paths = yaml.safe_load(yamlfile)

if os.path.isfile(OUT_JSON):
    with open(OUT_JSON) as jsonfile:
        out = json.load(jsonfile.read())
else:
    out = {}

for start_id, path_dict in paths.items():
    if start_id not in out.keys():
        logging.info('{}:'.format(start_id))
        paths_lst = []
        for tt in ['static', 'uniform', 'band structure']:
            paths = [p for p in path_dict[tt] if os.path.isdir(p)]
            paths_lst.extend(zip([tt] * len(paths), paths))
        for task_type, path in paths_lst:
            try:
                result = bader_analysis_from_path(path)
                out.update({
                    start_id: {
                        'task_type': task_type,
                        'path': path,
                        'result': result
                    }
                })
                logging.info('  Done within {} path:\n  {}\n'.format(task_type, path))
                break
            except:
                with open(OUT_JSON, 'w') as jsonfile:
                    json.dump(out, jsonfile, indent=4, separators=(',', ': '))
        else:
            logging.warn('  Vaild path not found.\n')

with open(OUT_JSON, 'w') as jsonfile:
    json.dump(out, jsonfile, indent=4, separators=(',', ': '))
