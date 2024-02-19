import platform
import re
import subprocess

from global_log import GlobalLog


def kill_udacity_simulator() -> None:
    udacity_program_name = "self_driving_car_nanodegree_program"
    _kill_process(program_name=udacity_program_name)


def kill_donkey_simulator() -> None:
    donkey_program_name = "donkey_sim"
    _kill_process(program_name=donkey_program_name)


def _kill_process(program_name: str) -> None:
    logg = GlobalLog(f"kill_{program_name}")
    plt = platform.system()
    assert plt.lower() == "windows", "Platform {} not supported yet".format(plt.lower())

    cmd = "tasklist"

    ret = subprocess.check_output(cmd)
    output_str = ret.decode("utf-8")

    # from https://stackoverflow.com/questions/13525882/tasklist-output
    tasks = output_str.split("\r\n")
    killed = False
    for task in tasks:
        m = re.match("(.+?) +(\d+) (.+?) +(\d+) +(\d+.* K).*", task)
        if m is not None:
            image = m.group(1)
            if program_name in image or image in program_name:
                cmd = 'taskkill /IM "{}.exe" /F'.format(program_name)
                ret = subprocess.check_output(cmd)
                output_str = ret.decode("utf-8")
                logg.info(output_str)
                killed = True

    if not killed:
        logg.warn(
            "The program {} is not in the list of currently running programs".format(
                program_name
            )
        )


def kill_beamng_simulator() -> None:
    beamng_program_name = "BeamNG.drive.x64"
    _kill_process(program_name=beamng_program_name)


if __name__ == "__main__":
    kill_udacity_simulator()
