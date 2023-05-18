from .run import run as default_run
from .on_off_run import run as on_off_run
from .dop_run import run as dop_run
from .per_run import run as per_run
from .gire_run import run as gire_run
from .snc_run import run as snc_run
from .gfootball_run import run as gfootball_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run
REGISTRY["gire_run"] = gire_run
REGISTRY["snc_run"] = snc_run
REGISTRY["gfootball_run"] = gfootball_run