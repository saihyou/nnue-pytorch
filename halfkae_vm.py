import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 5 * 9
NUM_PLANES = 1548 + 81 * 2
NUM_PLANES_FACT = NUM_PLANES
REL_FEATURES = 5870


class Features(FeatureBlock):
    def __init__(self):
        super(Features, self).__init__(
            "HalfKAEVm",
            0x5F134CB8,
            OrderedDict(
                [("HalfKAEVm", NUM_PLANES * NUM_SQ + NUM_PLANES * NUM_SQ * 3)]
            ),
        )

    def get_active_features(self, board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for factorizer support during training"
        )


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super(FactorizedFeatures, self).__init__(
            "HalfKAEVm^",
            0x5F134CB8,
            OrderedDict(
                [
                    ("HalfKAEVm", NUM_PLANES * NUM_SQ + NUM_PLANES * NUM_SQ * 3),
                    ("HalfKAVm", NUM_PLANES * NUM_SQ),
                    ("A", NUM_PLANES_FACT),
                    ("HalfRelKA", REL_FEATURES),
                ]
            ),
        )

    def get_active_features(self, board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for factorizer support during training"
        )

    def get_feature_factors(self, idx):
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")
        ka_index = idx
        while ka_index > NUM_PLANES * NUM_SQ:
            ka_index -= (NUM_PLANES - 90) * NUM_SQ
        k_idx = ka_index // NUM_PLANES
        a_idx = ka_index % NUM_PLANES

        def _make_relka_index(sq_k, p):
            if p < 90:
                return p
            w = 9 * 2 - 1
            h = 9 * 2 - 1
            piece_index = (p - 90) // 81
            sq_p = (p - 90) % 81
            relative_file = (sq_p // 9) - (sq_k // 9) + (w // 2)
            relative_rank = (sq_p % 9) - (sq_k % 9) + (h // 2)
            return int(h * w * piece_index + h * relative_file + relative_rank + 90)

        return [
            idx,
            self.get_factor_base_feature("HalfKAVm") + ka_index,
            self.get_factor_base_feature("A") + a_idx,
            self.get_factor_base_feature("HalfRelKA") + _make_relka_index(k_idx, a_idx),
        ]


def get_feature_block_clss():
    """
    This is used by the features module for discovery of feature blocks.
    """
    return [Features, FactorizedFeatures]
