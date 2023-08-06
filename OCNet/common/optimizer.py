import torch.optim as optim


def build_optimizer(_cfg, model):
    opt = _cfg._dict['OPTIMIZER']['NAME']
    lr = _cfg._dict['OPTIMIZER']['LEARNING_RATE']
    wd = _cfg._dict['OPTIMIZER']['WEIGHT_DECAY']

    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.module.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError('Optimizer {} not supported.'.format(opt))

    return optimizer


def build_scheduler(_cfg, optimizer):

  # Constant learning rate
  if _cfg._dict['SCHEDULER']['TYPE'] == 'constant':
    lambda1 = lambda epoch: 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

  # Learning rate scaled by 0.98^(epoch)
  if _cfg._dict['SCHEDULER']['TYPE'] == 'power_iteration':
    lambda1 = lambda epoch: (0.98) ** (epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


  return scheduler