# Frozen observations

`data.pkl`, `var.pkl`, `assim_index.csv`, `datatyp.csv` and `reportdates.csv`
are copied unchanged from the four runs behind the chapter figures
(`HPC:/cluster/work/users/krfo/DROGON/{ES-MDA-SIM2SEIS,ES-MDA-loc-SIM2SEIS,
ML_ES-MDA-SIM2SEIS,ML-SEQ_ES-MDA-SIM2SEIS}`), which all used the same files.
The three acoustic-impedance vintages are resized to the grid, scaled to
`[-1, 1]` and reversed along J, as described in Chapter 5.1. The setup scripts
do not import SEG-Y files or call the original observation-generation program.
Preserve the pickle files as the frozen inference inputs.
