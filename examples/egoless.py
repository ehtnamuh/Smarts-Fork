import logging

import gym

from smarts.core.utils.episodes import episodes

# The following ugliness was made necessary because the `aiohttp` #
# dependency has an "examples" module too.  (See PR #1120.)
if __name__ == "__main__":
    from argument_parser import default_argument_parser
else:
    from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        sim_name=sim_name,
        headless=headless,
        sumo_headless=True,
        visdom=False,
        seed=seed,
        fixed_timestep_sec=0.1,
    )

    if max_episode_steps is None:
        max_episode_steps = 1000

    for episode in episodes(n=num_episodes):
        env.reset()
        episode.record_scenario(env.scenario_log)

        for _ in range(max_episode_steps):
            env.step({})
            episode.record_step({}, {}, {}, {})

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("egoless-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
