# Camera-model algorithm attribution

`backend/camera_model.py` adapts the numerical projection from
`cmod_cahvore_3d_to_2d_general` in NASA-AMMOS VICAR, by Todd Litwin.

Original copyright: 1998, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2009,
2010, 2011, 2012, 2015, 2016, 2018, 2022 California Institute of Technology.
All Rights Reserved. NASA-AMMOS distributes VICAR under Apache License 2.0.

Source: https://github.com/NASA-AMMOS/VICAR/blob/master/vos/p2/sub/cahvor/cmod_cahvore.c

Changes: vectorized NumPy implementation, omitted derivatives, explicit invalid-ray
rejection, and PDS rover/site frame adapter. No original calibration values are invented.
The original project license is available at https://github.com/NASA-AMMOS/VICAR/blob/master/LICENSE.

NASA/JPL-Caltech PDS imagery and PLACES/HiRISE data retain the source credits and
provenance in the prepared dataset. Software and observations do not imply NASA endorsement.
