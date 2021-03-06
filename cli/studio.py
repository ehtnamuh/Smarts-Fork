# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from threading import Thread

import click


@click.group(name="scenario")
def scenario_cli():
    pass


@scenario_cli.command(name="build", help="Generate a single scenario")
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Clean previously generated artifacts first",
)
@click.option(
    "--allow-offset-map",
    is_flag=True,
    default=False,
    help="Allows Sumo's road network (map.net.xml) to be offset from the origin. if not specified, creates a network file that ends with AUTOGEN.net.xml if necessary.",
)
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def build_scenario(clean, allow_offset_map, scenario):
    _build_single_scenario(clean, allow_offset_map, scenario)


def _build_single_scenario(clean, allow_offset_map, scenario):
    click.echo(f"build-scenario {scenario}")
    if clean:
        _clean(scenario)

    scenario_root = Path(scenario)
    map_glb = scenario_root / "map.glb"
    od_map_paths = list(scenario_root.rglob("*.xodr"))
    sumo_map_paths = list(scenario_root.rglob("*.net.xml"))
    if od_map_paths:
        from smarts.sstudio.types import MapSpec
        from smarts.core.opendrive_road_network import OpenDriveRoadNetwork

        assert len(od_map_paths) == 1
        map_xodr = str(od_map_paths[0])
        map_spec = MapSpec(map_xodr)
        od_road_network = OpenDriveRoadNetwork.from_spec(map_spec)
        od_road_network.to_glb(str(map_glb))

    elif sumo_map_paths:
        from smarts.sstudio.sumo2mesh import generate_glb_from_sumo_file
        from smarts.core.sumo_road_network import SumoRoadNetwork

        map_net = None
        for map_path in sumo_map_paths:
            if not str(map_path).endswith("AUTOGEN.net.xml"):
                map_net = str(map_path)
                break

        assert map_net is not None
        if not allow_offset_map or scenario.traffic_histories:
            from smarts.sstudio.types import MapSpec

            map_spec = MapSpec(map_net)
            SumoRoadNetwork.from_spec(map_spec, shift_to_origin=True)
        elif os.path.isfile(SumoRoadNetwork.shifted_net_file_path(map_net)):
            click.echo(
                "WARNING: {} already exists.  Remove it if you want to use unshifted/offset map.net.xml instead.".format(
                    SumoRoadNetwork.shifted_net_file_name
                )
            )
        generate_glb_from_sumo_file(map_net, str(map_glb))
    else:
        click.echo(
            "FILENOTFOUND: no reference to network file was found in {}.  "
            "Please make sure the path passed is a valid Scenario with RoadNetwork file (map.net.xml or map.xodr) required "
            "for scenario building.".format(str(scenario_root))
        )
        return

    _install_requirements(scenario_root)

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        subprocess.check_call([sys.executable, scenario_py])


def _install_requirements(scenario_root):
    import importlib.resources as pkg_resources

    requirements_txt = scenario_root / "requirements.txt"
    if requirements_txt.exists():
        import zoo.policies

        with pkg_resources.path(zoo.policies, "") as path:
            # Serve policies through the static file server, then kill after
            # we've installed scenario requirements
            pip_index_proc = subprocess.Popen(
                ["twistd", "-n", "web", "--path", path],
                # Hide output to keep display simple
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            pip_install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_txt),
            ]

            click.echo(
                f"Installing scenario dependencies via '{' '.join(pip_install_cmd)}'"
            )

            try:
                subprocess.check_call(pip_install_cmd, stdout=subprocess.DEVNULL)
            finally:
                pip_index_proc.terminate()
                pip_index_proc.wait()


@scenario_cli.command(
    name="build-all",
    help="Generate all scenarios under the given directories",
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Clean previously generated artifacts first",
)
@click.option(
    "--allow-offset-maps",
    is_flag=True,
    default=False,
    help="Allows Sumo's road networks (map.net.xml) to be offset from the origin. if not specified, creates creates a network file that ends with AUTOGEN.net.xml' if necessary.",
)
@click.argument("scenarios", nargs=-1, metavar="<scenarios>")
def build_all_scenarios(clean, allow_offset_maps, scenarios):
    if not scenarios:
        # nargs=-1 in combination with a default value is not supported
        # if scenarios is not given, set /scenarios as default
        scenarios = ["scenarios"]

    builder_threads = {}
    for scenarios_path in scenarios:
        path = Path(scenarios_path)

        for p in path.rglob("*.xodr"):
            scenario = f"{scenarios_path}/{p.parent.relative_to(scenarios_path)}"
            builder_thread = Thread(
                target=_build_single_scenario, args=(clean, False, scenario)
            )
            builder_thread.start()
            builder_threads[p] = builder_thread

        for p in path.rglob("*map.net.xml"):
            scenario = f"{scenarios_path}/{p.parent.relative_to(scenarios_path)}"
            if scenario == f"{scenarios_path}/waymo":
                continue
            builder_thread = Thread(
                target=_build_single_scenario,
                args=(clean, allow_offset_maps, scenario),
            )
            builder_thread.start()
            builder_threads[p] = builder_thread

    for scenario_path, builder_thread in builder_threads.items():
        click.echo(f"Waiting on {scenario_path} ...")
        builder_thread.join()


@scenario_cli.command(name="clean")
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def clean_scenario(scenario):
    _clean(scenario)


def _clean(scenario):
    to_be_removed = [
        "map.glb",
        "bubbles.pkl",
        "missions.pkl",
        "flamegraph-perf.log",
        "flamegraph.svg",
        "flamegraph.html",
        "*.rou.xml",
        "*.rou.alt.xml",
        "social_agents/*",
        "traffic/*",
        "history_mission.pkl",
        "*.shf",
        "*-AUTOGEN.net.xml",
    ]
    p = Path(scenario)
    for file_name in to_be_removed:
        for f in p.glob(file_name):
            # Remove file
            f.unlink()


@scenario_cli.command(name="replay")
@click.option("-d", "--directory", multiple=True)
@click.option("-t", "--timestep", default=0.01, help="Timestep in seconds")
@click.option("--endpoint", default="ws://localhost:8081")
def replay(directory, timestep, endpoint):
    from envision.client import Client as Envision

    for path in directory:
        jsonl_paths = list(Path(path).glob("*.jsonl"))
        click.echo(
            f"Replaying {len(jsonl_paths)} record(s) at path={path} with "
            f"timestep={timestep}s"
        )

        with multiprocessing.pool.ThreadPool(len(jsonl_paths)) as pool:
            pool.starmap(
                Envision.read_and_send,
                [(jsonl, endpoint, timestep) for jsonl in jsonl_paths],
            )


scenario_cli.add_command(build_scenario)
scenario_cli.add_command(build_all_scenarios)
scenario_cli.add_command(clean_scenario)
scenario_cli.add_command(replay)
