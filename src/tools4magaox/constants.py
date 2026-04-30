# constants.py
# 2026/04/06
# constants that I end up using everhere

RAD_TO_AS = 206265.0
WFS_WAVELENGTH = 910e-9 # m

# this is also known as spatal resolution
LAM_D_rad = WFS_WAVELENGTH / 6.5 # rad
LAM_D_AS = LAM_D_rad * RAD_TO_AS # as

# platescale for camscis
CS_PLATESCALE = 0.00579 # as/px
CS_PS_LamD = CS_PLATESCALE / LAM_D_AS  # LD/px

# camsci properties
CAMSCI_FOV_PX = 1024 # px
CAMSCI_FOV_AS = CAMSCI_FOV_PX * CS_PLATESCALE #as
CAMSCI_PX_M = 13.7e-6 # m/px
CAMSCI_PX_AS = 0.00579 # as/px
CAMSCI_BITDEPTH = 16