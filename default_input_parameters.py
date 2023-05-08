import pyarrow.feather as feather


catchments_attributes_filename: str = "data_input/attributes_catchments_ChileCentral.feather"
attributes_catchments = feather.read_feather(catchments_attributes_filename)
cod_cuencas = attributes_catchments.cod_cuenca.to_list()

months_initialisation = ['may','jun','jul','ago','sep','oct','nov','dic','ene','feb']#,'mar']
