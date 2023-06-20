# check if the task needed to run
def if_run_task(tasklist: str, moduleName: str):
    if tasklist and moduleName in tasklist.split(","):
        return True
    elif not tasklist or moduleName not in tasklist.split(","):
        return False

