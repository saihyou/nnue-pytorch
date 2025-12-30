from collections import OrderedDict

from feature_block import FeatureBlock


NUM_SQ = 81
NUM_PLANES = 1548 + 81
NUM_KING_BUCKETS = 45
NUM_INPUTS = NUM_PLANES * NUM_KING_BUCKETS
REL_FEATURES = 5870


class Features(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKAv2_hm",
            0x7F134CB8,
            OrderedDict([("HalfKAv2_hm", NUM_INPUTS)]),
        )

    def get_active_features(self, board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for support during training"
        )


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super().__init__(
            "HalfKAv2_hm^",
            0x7F134CB8,
            OrderedDict(
                [
                    ("HalfKAv2_hm", NUM_INPUTS),
                    ("HalfRelKA", REL_FEATURES),
                ]
            ),
        )

    def get_active_features(self, board):
        raise Exception(
            "Not supported yet, you must use the c++ data loader for support during training"
        )

    def get_feature_factors(self, idx):
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        k_idx = idx // NUM_PLANES
        a_idx = idx % NUM_PLANES
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
        return [idx, self.get_factor_base_feature("HalfRelKA") + _make_relka_index(k_idx, a_idx)]


def get_feature_block_clss():
    return [Features, FactorizedFeatures]
