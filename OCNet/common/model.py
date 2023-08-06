from OCNet.models.LMSCNet import LMSCNet
from OCNet.models.LMSCNet_SS import LMSCNet_SS
from OCNet.models.SSCNet_full import SSCNet_full
from OCNet.models.SSCNet import SSCNet
from OCNet.models.OCNet import OCNet



def get_model(_cfg, dataset):

  nbr_classes = dataset.nbr_classes
  grid_dimensions = dataset.grid_dimensions
  class_frequencies = dataset.class_frequencies

  selected_model = _cfg._dict['MODEL']['TYPE']
  
  # OCNet -------------------------------------------------------------------------------------------------------
  if selected_model == 'OCNet':
    model = OCNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  # ------------------------------------------------------------------------------------------------------------------

  # LMSCNet ----------------------------------------------------------------------------------------------------------
  elif selected_model == 'LMSCNet':
    model = LMSCNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  # ------------------------------------------------------------------------------------------------------------------

  # LMSCNet_SS -------------------------------------------------------------------------------------------------------
  elif selected_model == 'LMSCNet_SS':
    model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  # ------------------------------------------------------------------------------------------------------------------
  
  # SSCNet_full ------------------------------------------------------------------------------------------------------
  elif selected_model == 'SSCNet_full':
    model = SSCNet_full(class_num=nbr_classes)
  # ------------------------------------------------------------------------------------------------------------------

  # SSCNet -----------------------------------------------------------------------------------------------------------
  elif selected_model == 'SSCNet':
    model = SSCNet(class_num=nbr_classes)
  # ------------------------------------------------------------------------------------------------------------------

  else:
    assert False, 'Wrong model selected'

  return model