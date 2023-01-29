import platform
import subprocess

from global_log import GlobalLog


def kill_beamng_simulator() -> None:

    logg = GlobalLog("kill_beamng_simulator")

    beamng_program_name = "BeamNG.drive.x64"

    plt = platform.system()
    assert plt.lower() == "windows", "Platform {} not supported yet".format(plt.lower())

    cmd = "tasklist"

    ret = subprocess.check_output(cmd)
    output_str = ret.decode("utf-8")

    program_name = beamng_program_name
    if program_name in output_str:
        cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
        ret = subprocess.check_output(cmd)
        output_str = ret.decode("utf-8")
        logg.info(output_str)
    else:
        logg.warn("The program {} is not in the list of currently running programs".format(beamng_program_name))
