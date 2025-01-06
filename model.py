import chess
import ranger
import ranger21
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
from feature_transformer import DoubleFeatureTransformerSlice

# 3 layer fully connected network
L1 = 1536
L2 = 15
L3 = 64

def coalesce_ft_weights(model, layer):
  weight = layer.weight.data
  indices = model.feature_set.get_virtual_to_real_features_gather_indices()
  weight_coalesced = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
  for i_real, is_virtual in enumerate(indices):
    weight_coalesced[i_real, :] = sum(weight[i_virtual, :] for i_virtual in is_virtual)
  return weight_coalesced

def get_parameters(layers):
  return [p for layer in layers for p in layer.parameters()]

class LayerStacks(nn.Module):
  def __init__(self, count):
    super(LayerStacks, self).__init__()

    self.count = count
    self.l1 = nn.Linear(2 * L1 // 2, (L2 + 1) * count)
    # Factorizer only for the first layer because later
    # there's a non-linearity and factorization breaks.
    # This is by design. The weights in the further layers should be
    # able to diverge a lot.
    self.l1_fact = nn.Linear(2 * L1 // 2, L2 + 1, bias=True)
    self.l2 = nn.Linear(L2*2, L3 * count)
    self.output = nn.Linear(L3, 1 * count)

    # Cached helper tensor for choosing outputs by bucket indices.
    # Initialized lazily in forward.
    self.idx_offset = None

    self._init_layers()

  def _init_layers(self):
    l1_weight = self.l1.weight
    l1_bias = self.l1.bias
    l1_fact_weight = self.l1_fact.weight
    l1_fact_bias = self.l1_fact.bias
    l2_weight = self.l2.weight
    l2_bias = self.l2.bias
    output_weight = self.output.weight
    output_bias = self.output.bias
    with torch.no_grad():
      l1_fact_weight.fill_(0.0)
      l1_fact_bias.fill_(0.0)
      output_bias.fill_(0.0)

      for i in range(1, self.count):
        # Force all layer stacks to be initialized in the same way.
        l1_weight[i*(L2+1):(i+1)*(L2+1), :] = l1_weight[0:(L2+1), :]
        l1_bias[i*(L2+1):(i+1)*(L2+1)] = l1_bias[0:(L2+1)]
        l2_weight[i*L3:(i+1)*L3, :] = l2_weight[0:L3, :]
        l2_bias[i*L3:(i+1)*L3] = l2_bias[0:L3]
        output_weight[i:i+1, :] = output_weight[0:1, :]

    self.l1.weight = nn.Parameter(l1_weight)
    self.l1.bias = nn.Parameter(l1_bias)
    self.l1_fact.weight = nn.Parameter(l1_fact_weight)
    self.l1_fact.bias = nn.Parameter(l1_fact_bias)
    self.l2.weight = nn.Parameter(l2_weight)
    self.l2.bias = nn.Parameter(l2_bias)
    self.output.weight = nn.Parameter(output_weight)
    self.output.bias = nn.Parameter(output_bias)

  def forward(self, x, ls_indices):
    # Precompute and cache the offset for gathers
    if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
      self.idx_offset = torch.arange(0,x.shape[0]*self.count,self.count, device=ls_indices.device)

    indices = ls_indices.flatten() + self.idx_offset

    l1s_ = self.l1(x).reshape((-1, self.count, L2 + 1))
    l1f_ = self.l1_fact(x)
    # https://stackoverflow.com/questions/55881002/pytorch-tensor-indexing-how-to-gather-rows-by-tensor-containing-indices
    # basically we present it as a list of individual results and pick not only based on
    # the ls index but also based on batch (they are combined into one index)
    l1c_ = l1s_.view(-1, L2 + 1)[indices]
    l1c_, l1c_out = l1c_.split(L2, dim=1)
    l1f_, l1f_out = l1f_.split(L2, dim=1)
    l1x_ = l1c_ + l1f_
    # multiply sqr crelu result by (127/128) to match quantized version
    l1x_ = torch.clamp(torch.cat([torch.pow(l1x_, 2.0) * (127/128), l1x_], dim=1), 0.0, 1.0)

    l2s_ = self.l2(l1x_).reshape((-1, self.count, L3))
    l2c_ = l2s_.view(-1, L3)[indices]
    l2x_ = torch.clamp(l2c_, 0.0, 1.0)

    l3s_ = self.output(l2x_).reshape((-1, self.count, 1))
    l3c_ = l3s_.view(-1, 1)[indices]
    l3x_ = l3c_ + l1f_out + l1c_out

    return l3x_

  def get_coalesced_layer_stacks(self):
    # During training the buckets are represented by a single, wider, layer.
    # This representation needs to be transformed into individual layers
    # for the serializer, because the buckets are interpreted as separate layers.
    for i in range(self.count):
      with torch.no_grad():
        l1 = nn.Linear(2*L1 // 2, L2 + 1)
        l2 = nn.Linear(L2*2, L3)
        output = nn.Linear(L3, 1)
        l1.weight.data = self.l1.weight[i*(L2+1):(i+1)*(L2+1), :] + self.l1_fact.weight.data
        l1.bias.data = self.l1.bias[i*(L2+1):(i+1)*(L2+1)] + self.l1_fact.bias.data
        l2.weight.data = self.l2.weight[i*L3:(i+1)*L3, :]
        l2.bias.data = self.l2.bias[i*L3:(i+1)*L3]
        output.weight.data = self.output.weight[i:(i+1), :]
        output.bias.data = self.output.bias[i:(i+1)]
        yield l1, l2, output

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, start_lambda=1.0, end_lambda=1.0, max_epoch=800, gamma=0.992, lr=8.75e-4, epoch_size=100_000_000, batch_size=16384, in_scaling=240, out_scaling=280, offset=270, adjust_loss=0.1):
    super(NNUE, self).__init__()
    self.num_ls_buckets = 4
    self.input = DoubleFeatureTransformerSlice(feature_set.num_features, L1)
    self.feature_set = feature_set
    self.layer_stacks = LayerStacks(self.num_ls_buckets)
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
    self.adjust_loss = adjust_loss
  
    max_hidden_weight = self.quantized_one / self.weight_scale_hidden
    max_out_weight = (self.quantized_one * self.quantized_one) / (self.nnue2score * self.weight_scale_out)
    self.weight_clipping = [
      {'params' : [self.layer_stacks.l1.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight, 'virtual_params' : self.layer_stacks.l1_fact.weight },
      {'params' : [self.layer_stacks.l2.weight], 'min_weight' : -max_hidden_weight, 'max_weight' : max_hidden_weight },
      {'params' : [self.layer_stacks.output.weight], 'min_weight' : -max_out_weight, 'max_weight' : max_out_weight },
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
        weights[a:b, :] = 0.0
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
      padding = weights.new_zeros((new_feature_block.num_virtual_features, weights.shape[1]))
      weights = torch.cat([weights, padding], dim=0)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices):
    w, b = self.input(white_indices, white_values, black_indices, black_values)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l0_s = torch.split(l0_, L1 // 2, dim=1)
    l0_s1 = [l0_s[0] * l0_s[1], l0_s[2] * l0_s[3]]
    # We multiply by 127/128 because in the quantized network 1.0 is represented by 127
    # and it's more efficient to divide by 128 instead.
    l0_ = torch.cat(l0_s1, dim=1) * (127/128)
    x = self.layer_stacks(l0_, layer_stack_indices)
    return x

  def step_(self, batch, batch_idx, loss_type):
    self._clip_weights()
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, layer_stack_indices = batch

    # convert the network and search scores to an estimate match result
    # based on the win_rate_model, with scalings and offsets optimized
    in_scaling = self.in_scaling
    out_scaling = self.out_scaling
    offset = self.offset

    scorenet = self(us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices) * self.nnue2score
    q  = ( scorenet - offset) / in_scaling  # used to compute the chance of a win
    qm = (-scorenet - offset) / in_scaling  # used to compute the chance of a loss
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())  # estimated match result (using win, loss and draw probs).

    p  = ( score - offset) / out_scaling
    pm = (-score - offset) / out_scaling
    pf = 0.5 * (1.0 + p.sigmoid() - pm.sigmoid())

    t = outcome
    actual_lambda = self.start_lambda + (self.end_lambda - self.start_lambda) * (self.current_epoch / self.max_epoch)
    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    loss = torch.pow(torch.abs(pt - qf), 2.5)
    loss = loss * ((qf > pt) * self.adjust_loss + 1)
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
      {'params' : get_parameters([self.input]), 'lr' : LR, 'gc_dim' : 0 },
      {'params' : [self.layer_stacks.l1_fact.weight], 'lr' : LR },
      {'params' : [self.layer_stacks.l1_fact.bias], 'lr' : LR },
      {'params' : [self.layer_stacks.l1.weight], 'lr' : LR },
      {'params' : [self.layer_stacks.l1.bias], 'lr' : LR },
      {'params' : [self.layer_stacks.l2.weight], 'lr' : LR },
      {'params' : [self.layer_stacks.l2.bias], 'lr' : LR },
      {'params' : [self.layer_stacks.output.weight], 'lr' : LR },
      {'params' : [self.layer_stacks.output.bias], 'lr' : LR },
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
