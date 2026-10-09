<%!
import numpy as np
import re
from scipy.stats import norm
from scipy import sparse
level_path = ['../Levels/Level300/','../Levels/Level9000/','../Levels/Level17500/', '../Levels/Level104098/']

def _extract_satnum(prob,g):
    unif_g = norm.cdf(g)
    prob_fields = np.stack([prob[f'p{i}'].reshape(unif_g.shape) for i in [1,2,3,4,7,8,9,10,11,12]], axis=-1)
    cumulative_probs = np.cumsum(prob_fields, axis=-1)

    # Fully vectorized assignment using broadcasting
    # Create comparison array: unif_g[..., None] vs cumulative_probs
    unif_expanded = unif_g[..., np.newaxis]
    comparisons = unif_expanded <= cumulative_probs

    # Find first True along the last axis (first threshold exceeded)
    satnum_indices = np.argmax(comparisons, axis=-1)

    # Handle case where no threshold is exceeded (should get SATNUM 12)
    no_threshold_exceeded = ~np.any(comparisons, axis=-1)
    satnum_indices[no_threshold_exceeded] = 11  # Will map to SATNUM 12

    # Convert indices to SATNUM values (1-12)
    satnum = (satnum_indices + 1).astype(np.int32)
    satnum[no_threshold_exceeded] = 12

    # Handle inactive cells efficiently
    inactive_mask = prob['p1'] < 0
    if inactive_mask.shape != satnum.shape:
        inactive_mask = inactive_mask.reshape(satnum.shape)
    satnum[inactive_mask] = 12

    return satnum
%>

<%
dim = int(re.search(r'Level(\d+)', level_path[level]).group(1))
if level != 3:
    T = sparse.load_npz(f'{level_path[level]}TransformMatMean.npz')
    permx_val = T.dot(permx)
    permy_val = T.dot(permy)
    permz_val = T.dot(permz)
    poro_val = T.dot(poro)
    satnum_tmp = T.dot(satnum)
else:
    permx_val = permx
    permy_val = permy
    permz_val = permz
    poro_val = poro
    satnum_tmp = satnum

prob = np.load(f'{level_path[level]}probfaction.npz', allow_pickle=True)
satnum_val = _extract_satnum(prob, satnum_tmp)
region_files = {name: f'../{level_path[level]}{name}.grdecl'
                for name in ('MULTNUM', 'EQLNUM', 'FIPNUM', 'FIPZON', 'PVTNUM')}
if level == 3:
    region_files = {
        'MULTNUM': '../../include/grid/drogon.multnum',
        'EQLNUM': '../../include/regions/drogon.eqlnum',
        'FIPNUM': '../../include/regions/drogon.fipnum',
        'FIPZON': '../../include/regions/drogon.fipzon',
        'PVTNUM': '../../include/regions/drogon.pvtnum',
    }
%>

-- This reservoir simulation deck is made available under the Open Database
-- License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in
-- individual contents of the database are licensed under the Database Contents
-- License: http://opendatacommons.org/licenses/dbcl/1.0/

-- Copyright (C) 2022 Equinor

--==============================================================================
--		Synthetic reservoir simulation model Drogon (2020)
--==============================================================================

-- The Drogon model - the successor of the Reek model
-- Used in a FMU set up
-- Grid input data generated from RMS project 


-- =============================================================================
RUNSPEC
-- =============================================================================

-- Simulation run title
TITLE
 Drogon synthetic reservoir model

-- Simulation run start
START
 1 JAN 2018 /

-- Fluid phases present
OIL
GAS
WATER
DISGAS
VAPOIL

-- Measurement unit used
METRIC

-- Options for equilibration
EQLOPTS
 'THPRES'  /

-- Dimensions and options for tracers
-- 2 water tracers
TRACERS
 1*  2 /

-- Grid dimension
INCLUDE
  '../../include/runspec/drogon.dimens' / -- exported by rms

-- Table dimensions
INCLUDE
  '../../include/runspec/drogon.tabdims' / -- exported by rms

-- Dimension of equilibration tables
INCLUDE
  '../../include/runspec/drogon.eqldims' / -- exported by rms

-- Regions dimension data
INCLUDE
  '../../include/runspec/drogon.regdims' / -- exported by rms

-- x-,y-,z- and multnum regions
INCLUDE
  '../../include/runspec/drogon.gridopts' / -- exported by rms

-- Dimensions for fault data
FAULTDIM
 500 /

-- Well dimension data
-- nwmaxz: max wells in the model
-- ncwmax: max connections per well
-- ngmaxz: max groups in the model
-- nwgmax: max wells in any one group
WELLDIMS
-- nwmaxz  ncwmax  ngmaxz  nwgmax
   20       100     10       20 /

-- Dimensions for multi-segment wells
-- nswlmx: max multi-segment wells in the model
-- nsegmx: max segments per well
-- nlbrmx: max branches per multi-segment well
WSEGDIMS
-- nswlmx  nsegmx  nlbrmx
   3       150      100 /


-- Input and output files format
UNIFIN
UNIFOUT

-- Disables the initial index file output
NOINSPEC

-- Disables the restart index file output
NORSSPEC


-- print and stop limits
-- -----------print------------  -----------stop--------------------
-- mes  com  war  prb  err  bug  mes  com   war     prb    err  bug  
MESSAGES
   1*   1*   1*   1000 10   1*   1*    1*  1000000 7000    0   /


-- =============================================================================
GRID
-- =============================================================================
NOECHO

NEWTRAN

GRIDFILE
 0 1 /

INIT

MINPV
* /

--Generates connections across pinched-out layers
PINCH
 3*  ALL  /

INCLUDE
${f"'../{level_path[level]}Grid.grdecl' /"} --exported by rms

INCLUDE
${f"'../{level_path[level]}FAULT.INC' /"} --exported by rms

PERMX
% for i in range(dim):
% if permx_val[i] > 8.5:
${"%.9f" %(np.exp(8.5))}
% elif permx_val[i] < -5:
${"%.9f" %(np.exp(-5))}
% else:
${"%.9f" %(np.exp(permx_val[i]))}
% endif
% endfor
/

PERMY
% for i in range(dim):
% if permy_val[i] > 8.5:
${"%.9f" %(np.exp(8.5))}
% elif permy_val[i] < -5:
${"%.9f" %(np.exp(-5))}
% else:
${"%.9f" %(np.exp(permy_val[i]))}
% endif
% endfor
/

PERMZ
% for i in range(dim):
% if permz_val[i] > 8.5:
${"%.9f" %(np.exp(8.5))}
% elif permz_val[i] < -5:
${"%.9f" %(np.exp(-5))}
% else:
${"%.9f" %(np.exp(permz_val[i]))}
% endif
% endfor
/

PORO
% for i in range(dim):
% if poro_val[i] > 0.5:
${"0.5"}
% elif poro_val[i] < 0.01:
${"0.01"}
% else:
${"%.9f" %(poro_val[i])}
% endif
% endfor
/
 
INCLUDE
${f"'{region_files['MULTNUM']}' /"}

INCLUDE
 '../../include/grid/drogon.multregt' / --from ert template

MULTFLT
${"'F1' {} /".format(norm.cdf(f1[0])*(1.2 - 0.0) + 0.0)}
${"'F2' {} /".format(norm.cdf(f2[0])*(1.2 - 0.0) + 0.0)}
${"'F3' {} /".format(norm.cdf(f3[0])*(1.2 - 0.0) + 0.0)}
${"'F4' {} /".format(norm.cdf(f4[0])*(1.2 - 0.0) + 0.0)}
${"'F5' {} /".format(norm.cdf(f5[0])*(1.2 - 0.0) + 0.0)}
${"'F6' {} /".format(norm.cdf(f6[0])*(1.2 - 0.0) + 0.0)}
/

-- =============================================================================
EDIT
-- =============================================================================

INCLUDE
${f"'../{level_path[level]}drogon_US.trans' /"}

-- =============================================================================
PROPS
-- =============================================================================

FILLEPS

INCLUDE                                
 '../../include/props/drogon.sattab' / --exported by rms

--INCLUDE
-- '../../include/props/drogon.swatinit' / --exported by rms

INCLUDE
 '../../include/props/drogon.pvt' /

-- Set up tracers
TRACER
 WT1  WAT 'g' /
 WT2  WAT 'g' /
/

EXTRAPMS
  4 /

-- =============================================================================
REGIONS
-- =============================================================================

INCLUDE
${f"'{region_files['EQLNUM']}' /"}

INCLUDE
${f"'{region_files['FIPNUM']}' /"}

INCLUDE
${f"'{region_files['FIPZON']}' /"}

SATNUM
% for val in satnum_val:
${f'{int(val)}'}
% endfor
/

-- INCLUDE
-- '../../include/regions/drogon.satnum' / --exported by rms

INCLUDE
${f"'{region_files['PVTNUM']}' /"}

-- =============================================================================
SOLUTION
-- =============================================================================
  
INCLUDE                                
 '../../include/solution/drogon.equil' / --exported by rms
INCLUDE
 '../../include/solution/drogon.rxvd' / --!! manually created (7 equil regions)
 
INCLUDE                                
 '../../include/solution/drogon.thpres' / --exported by rms

-- Initial tracer concentration vs depth for tracer WT1
TVDPFWT1
 1000  0.0 
 2500  0.0 /   

-- Initial tracer concentration vs depth for tracer WT2
TVDPFWT2
 1000  0.0 
 2500  0.0 /   

RPTSOL
 RESTART=2  FIP=2  'THPRES'  'FIPRESV' /

-- ALLPROPS --> fluid densities, viscosities , reciprocal formation volume factors and phase relative permeabilities
-- NORST=1  --> output for visualization only 
RPTRST
 ALLPROPS RVSAT RSSAT PBPD  NORST=1 RFIP RPORV /


-- =============================================================================
SUMMARY
-- =============================================================================

--RPTONLY

SUMTHIN
 1 /

INCLUDE
 '../../include/summary/drogon.summary' /


-- =============================================================================
SCHEDULE
-- =============================================================================

INCLUDE
${f"'../{level_path[level]}US_schdl.sch' /"}
