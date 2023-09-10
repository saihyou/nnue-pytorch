import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

NUM_SQ = 81
#NUM_PT = 12
#NUM_PLANES = 1629
NUM_PLANES = 1629 + 81
NUM_PLANES_FACT = NUM_PLANES
REL_FEATURES = 5870

def orient(is_white_pov: bool, sq: int):
  return (63 * (not is_white_pov)) ^ sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
  p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
  return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

def halfka_psqts():
  values = [0] * (NUM_PLANES * NUM_SQ)
  for ksq in range(81):
    for i in range(1, 20):
      values[NUM_PLANES * ksq + i] = 90
    for i in range(20, 39):
      values[NUM_PLANES * ksq + i] = -90
    for i in range(39, 44):
      values[NUM_PLANES * ksq + i] = 315
    for i in range(44, 49):
      values[NUM_PLANES * ksq + i] = -315
    for i in range(49, 54):
      values[NUM_PLANES * ksq + i] = 405
    for i in range(54, 59):
      values[NUM_PLANES * ksq + i] = -405
    for i in range(59, 64):
      values[NUM_PLANES * ksq + i] = 495
    for i in range(64, 69):
      values[NUM_PLANES * ksq + i] = -495
    for i in range(69, 74):
      values[NUM_PLANES * ksq + i] = 540
    for i in range(74, 79):
      values[NUM_PLANES * ksq + i] = -540
    for i in range(79, 82):
      values[NUM_PLANES * ksq + i] = 855
    for i in range(82, 85):
      values[NUM_PLANES * ksq + i] = -855
    for i in range(85, 88):
      values[NUM_PLANES * ksq + i] = 990
    for i in range(88, 90):
      values[NUM_PLANES * ksq + i] = -990
  index = 90
  for val in (90, -90, 315, -315, 405, -405, 495, -495, 540, -540, 855, -855, 990, -990):
    for ksq in range(81):
      for s in range(81):
        values[NUM_PLANES * ksq + index + s] = val
    index += 81

  return values

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ)]))

  def get_initial_psqt_features(self):
    return halfka_psqts()

  def get_active_features(self, board: chess.Board):
    def piece_features(turn):
      indices = torch.zeros(NUM_PLANES * NUM_SQ)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn)), sq, p)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA^', 0x5f134cb8, OrderedDict([('HalfKA', NUM_PLANES * NUM_SQ), ('A', NUM_PLANES_FACT), ('HalfRelKA', REL_FEATURES)]))

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

  def get_initial_psqt_features(self):
    return halfka_psqts() + [0] * (NUM_PLANES_FACT + REL_FEATURES)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
