from atomate.vasp.firetasks.parse_outputs import VaspToDb
db_file = '/lustre/home/acct-umjzhh/umjzhh-2/atomate/config/db.json'
a = VaspToDb(db_file=db_file,parse_dos=True,defuse_unsuccessful=True,additional_fields={"task_label":"static"})#{"task_label":"structure optimization"}#db_file=db_file,parse_dos=True,defuse_unsuccessful=True
fw_spec = ''#{"additional_fields":{"task_label":"static"}}#{"parse_dos":False,"defuse_unsuccessful":True,"db_file":db_file}#"additional_fields":{"task_label":"static"}
a.run_task(fw_spec)
