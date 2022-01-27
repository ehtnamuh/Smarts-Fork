from pathlib import Path

from smarts.core import seed
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

seed(42)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=1,
            actors={t.TrafficActor(name="car", vehicle_type=vehicle_type): 1},
        )
        for vehicle_type in [
            "trailer",
        ]
    ]
)

# laner_actor = t.SocialAgentActor(
#     name="keep-lane-agent",
#     agent_locator="zoo.policies:keep-lane-agent-v0",
# )

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
