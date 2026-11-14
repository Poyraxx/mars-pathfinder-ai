import numpy as np
import pytest
from backend.coordinates import wrap_lon,project,unproject,point_to_cell,cell_to_point,planetographic_to_centric,site_ned_to_enu

META={"radius_m":3396190.,"standard_parallel_deg":0,"central_lon_deg":0,"central_lat_deg":0,"west_m":8146952.49279,"north_m":-271815.07519,"cell_m":1}


def test_mars_roundtrip_and_grid_centres():
    for p in [{"lat":-4.59,"lon":137.448},{"lat":70,"lon":-170}]:
        x,y=project(**p,meta=META)
        restored=unproject(x,y,META)
        assert restored["lat"]==pytest.approx(p["lat"])
        assert restored["lon"]==pytest.approx(p["lon"])
    for rc in [(0,0),(511,511),(300,200)]:assert point_to_cell(cell_to_point(*rc,META),META)==rc


def test_longitude_convention_and_latitude_type():
    assert wrap_lon(360)==0
    assert wrap_lon(180)==-180
    assert wrap_lon(-181)==179
    assert planetographic_to_centric(45,3396190,3376200)<45


def test_ned_is_not_enu():
    assert np.array_equal(site_ned_to_enu(np.array([[1,2,3]])),[[2,1,-3]])
