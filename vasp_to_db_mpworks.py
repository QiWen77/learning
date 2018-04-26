from mpworks.firetasks.vasp_io_tasks import VaspCopyTask,VaspToDBTask
from mpworks.workflows.wf_utils import last_relax, get_loc, move_to_garden
import os 
import shutil
shutil.copyfile('/lustre/home/umjzhh-2/pv_me2/config/config_SjtuPi/my_launchpad.yaml','my_launchpad.yaml')
shutil.copyfile('/lustre/home/umjzhh-2/pv_me2/config/config_SjtuPi/FW_config.yaml','FW_config.yaml')
shutil.copyfile('/lustre/home/umjzhh-2/pv_me2/config/config_SjtuPi/my_fworker.yaml','my_fworker.yaml')
shutil.copyfile('/lustre/home/umjzhh-2/pv_me2/config/config_SjtuPi/my_qadapter.yaml','my_qadapter.yaml')
fw_spec = {"prev_vasp_dir":os.getcwd(),"prev_task_type":"GGA static v2","run_tags":"GGA static v2","dir_name":os.getcwd()}#GGA optimize structure (2x)"#"run_tags":"GGA static v2"#GGA static v2 #GGA optimize structure (2x)
copy = VaspCopyTask()
#copy.run_task(fw_spec)
parameters = {"additional_fields":{}}
to_db = VaspToDBTask()
to_db.run_task(fw_spec)

