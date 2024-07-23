import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 5 * 9
NUM_PLANES = 1548 + 81 * 2
NUM_PLANES_FACT = NUM_PLANES
REL_FEATURES = 5870


def orient(is_white_pov: bool, sq: int):
    return (63 * (not is_white_pov)) ^ sq


def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
    return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES


class Features(FeatureBlock):
    def __init__(self):
        super(Features, self).__init__("HalfKAVm", 0x7f234cb8, OrderedDict([("HalfKAVm", NUM_PLANES * NUM_SQ)]))

    def get_active_features(self, board: chess.Board):
        def piece_features(turn):
            indices = torch.zeros(NUM_PLANES * NUM_SQ)
            for sq, p in board.piece_map().items():
                indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
            return indices
        return (piece_features(chess.WHITE), piece_features(chess.BLACK))


class FactorizedFeatures(FeatureBlock):
    def __init__(self):
        super(FactorizedFeatures, self).__init__('HalfKAVm^', 0x7f234cb8, OrderedDict([('HalfKAVm', NUM_PLANES * NUM_SQ), ('A', NUM_PLANES_FACT), ('HalfRelKA', REL_FEATURES)]))

    def get_active_features(self, board: chess.Board):
        raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

    def get_feature_factors(self, idx):
        if idx >= self.num_real_features:
            raise Exception('Feature must be real')

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

        return [idx, self.get_factor_base_feature('A') + a_idx, self.get_factor_base_feature('HalfRelKA') + _make_relka_index(k_idx, a_idx)]


def get_feature_block_clss():
    '''
    This is used by the features module for discovery of feature blocks.
    '''
    return [Features, FactorizedFeatures]
