REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .gire_parallel_runner import GireParallelRunner
REGISTRY["gire_parallel"] = GireParallelRunner

from .snc_parallel_runner import SNCParallelRunner
REGISTRY["snc_parallel"] = SNCParallelRunner

from .gire_episode_runner import GireEpisodeRunner
REGISTRY["gire_episode"] = GireEpisodeRunner

from .gfootball_parallel_runner import GfootballParallelRunner
REGISTRY["gfootball_parallel"] = GfootballParallelRunner
