REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC

from .basic_herl_controller import BasicHerlMAC
from .n_herl_controller import NHERLMAC

from .basic_gire_controller import BasicGireMAC
from .n_gire_controller import NGIREMAC

from .basic_snc_controller import BasicSNCMAC
from .n_snc_controller import NSNCMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC

REGISTRY["herl_mac"] = BasicHerlMAC
REGISTRY["n_herl_mac"] = NHERLMAC

REGISTRY["gire_mac"] = BasicGireMAC
REGISTRY["n_gire_mac"] = NGIREMAC

REGISTRY["snc_mac"] = BasicSNCMAC
REGISTRY["n_snc_mac"] = NSNCMAC