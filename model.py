import chess
import ranger
import ranger21
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys

# 3 layer fully connected network
L1 = 1024
L2 = 8
L3 = 32

def get_parameters(layers):
  return [p for layer in layers for p in layer.parameters()]

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, start_lambda=1.0, end_lambda=1.0, max_epoch=800, gamma=0.992, lr=8.75e-4, epoch_size=100_000_000, batch_size=16384, in_scaling=240, out_scaling=280, offset=270):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.l1 = nn.Linear(L1, L2)
    self.l2 = nn.Linear(L2 * 2, L3)
    self.output = nn.Linear(L3, 1)
    self.start_lambda = start_lambda
    self.end_lambda = end_lambda
    self.gamma = gamma
    self.lr = lr
    self.nnue2score = 600.0
    self.weight_scale_hidden = 64.0
    self.weight_scale_out = 16.0
    self.quantized_one = 127.0
    self.max_epoch = max_epoch
    self.epoch_size = epoch_size
    self.batch_size = batch_size
    self.in_scaling = in_scaling
    self.out_scaling = out_scaling
    self.offset = offset
  
    max_hidden_weight = self.quantized_one / self.weight_scale_hidden
    max_out_weight = (self.quantized_one * self.quantized_one) / (self.nnue2score * self.weight_scale_out)
    self.weight_clipping = [
      {'params' : [self.l1.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.l2.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.output.weight], 'min_weight' : -max_out_weight, 'max_weight' : max_out_weight },
    ]

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    with torch.no_grad():
      for a, b in self.feature_set.get_virtual_feature_ranges():
        weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  Clips the weights of the model based on the min/max values allowed
  by the quantization scheme.
  '''
  def _clip_weights(self):
    for group in self.weight_clipping:
      for p in group['params']:
        if 'min_weight' in group or 'max_weight' in group:
          p_data_fp32 = p.data
          min_weight = group['min_weight']
          max_weight = group['max_weight']
          if 'virtual_params' in group:
            virtual_params = group['virtual_params']
            xs = p_data_fp32.shape[0] // virtual_params.shape[0]
            ys = p_data_fp32.shape[1] // virtual_params.shape[1]
            expanded_virtual_layer = virtual_params.repeat(xs, ys)
            if min_weight is not None:
              min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - expanded_virtual_layer
              p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
            if max_weight is not None:
              max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - expanded_virtual_layer
              p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
          else:
            if min_weight is not None and max_weight is not None:
              p_data_fp32.clamp_(min_weight, max_weight)
            else:
              raise Exception('Not supported.')
          p.data.copy_(p_data_fp32)


  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l0_s = torch.split(l0_, L1 // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
    # and it's more efficient to divide by 128 instead.
    l0_ = torch.cat(l0_s1, dim=1) * (127/128)
    l1x_ = self.l1(l0_)
    l1_ = torch.clamp(torch.cat([torch.pow(l1x_, 2.0) * (127/128), l1x_], dim=1), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def step_(self, batch, batch_idx, loss_type):
    self._clip_weights()
    us, them, white, black, outcome, score = batch

    # convert the network and search scores to an estimate match result
    # based on the win_rate_model, with scalings and offsets optimized
    in_scaling = self.in_scaling
    out_scaling = self.out_scaling
    offset = self.offset

    scorenet = self(us, them, white, black) * self.nnue2score
    q  = ( scorenet - offset) / in_scaling  # used to compute the chance of a win
    qm = (-scorenet - offset) / in_scaling  # used to compute the chance of a loss
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result (using win, loss and draw probs).

    p  = ( score - offset) / out_scaling
    pm = (-score - offset) / out_scaling
    pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

    t = outcome
    actual_lambda = self.start_lambda + (self.end_lambda - self.start_lambda) * (self.current_epoch / self.max_epoch)
    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    loss = torch.pow(torch.abs(pt - qf), 2.5).mean()
    loss = loss * ((qf > pt) * 0.1 + 1)
    loss = loss.mean()

    self.log(loss_type, loss)
    return loss

    # MSE Loss function for debugging
    # Scale score by 600.0 to match the expected NNUE scaling factor
    # output = self(us, them, white, black) * 600.0
    # loss = F.mse_loss(output, score)

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def configure_optimizers(self):
    LR = self.lr
    train_params = [
      {"params": get_parameters([self.input]), "lr": LR, "gc_dim": 0},
      {"params": [self.l1.weight], "lr": LR},
      {"params": [self.l1.bias], "lr": LR},
      {"params": [self.l2.weight], "lr": LR},
      {"params": [self.l2.bias], "lr": LR},
      {"params": [self.output.weight], "lr": LR},
      {"params": [self.output.bias], "lr": LR},
    ]
    # Increasing the eps leads to less saturated nets with a few dead neurons.
    # Gradient localisation appears slightly harmful.
    #optimizer = ranger.Ranger(
    #  train_params, betas=(0.9, 0.999), eps=1.0e-7, gc_loc=False, use_gc=False
    #)
    optimizer = ranger21.Ranger21(train_params,
      lr=1.0, betas=(.9, 0.999), eps=1.0e-7,
      using_gc=False, using_normgc=False,
      weight_decay=0.0,
      num_batches_per_epoch=int(self.epoch_size / self.batch_size), num_epochs=self.max_epoch,
      warmdown_active=False, use_warmup=False,
      use_adaptive_gradient_clipping=False,
      softplus=False,
      pnm_momentum_factor=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, step_size=1, gamma=self.gamma
    )
    return [optimizer], [scheduler]

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
