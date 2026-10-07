# Frozen observations

`data.pkl` and `var.pkl` are copied unchanged from the final normalized-input
variant at `Jupiter2:~/bookchapter/DROGON_w_seis/ML_ES-MDA-SIM2SEIS/`. This is
the Jupiter2 run variant whose three acoustic-impedance vintages are scaled to
`[-1, 1]`, as described in Chapter 5.1. It is used as the common observation
set for all four methods. The setup scripts do not import SEG-Y files or call
the original observation-generation program.

The accompanying date, index, and datatype files are from that same run
directory. Preserve the pickle files as the frozen inference inputs.
