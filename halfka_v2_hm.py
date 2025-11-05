from collections import OrderedDict

from feature_block import FeatureBlock


NUM_SQ = 81
NUM_PLANES = 1548 + 81
NUM_KING_BUCKETS = 45
NUM_INPUTS = NUM_PLANES * NUM_KING_BUCKETS


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
                    ("A", NUM_PLANES),
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

        plane = idx % NUM_PLANES
        return [idx, self.get_factor_base_feature("A") + plane]


def get_feature_block_clss():
    return [Features, FactorizedFeatures]
