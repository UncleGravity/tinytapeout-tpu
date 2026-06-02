# Multicycle constraint for the Q1A8 fp32 accumulator.
#
# q1a8_rowblock's per-lane accumulator register `acc_flat` only captures when
# `reducer_valid` pulses, which happens once per Q8 sub-block. The kernel FSM
# (q1a8_kernel.v) asserts rowblock_valid only in ST_ISSUE, and every ST_ISSUE
# is preceded by >= WBITS_BEATS (=4 for ROWS=8) weight-load cycles, so two
# consecutive accumulations into the same lane are ALWAYS >= 5 cycles apart
# (more under stream backpressure or across rowblock/scale boundaries).
#
# The combinational acc -> fp32_add -> acc loop therefore has >= 5 clock
# periods to settle, but the default single-cycle analysis of that loop
# reports a large false setup violation (~-14 ns at 7 ns). Declare the real
# >= 5-cycle relationship; 4 is conservative, hold = setup - 1.
#
# NOTE (XDC restriction): no `if`/`puts`/control flow allowed in an .xdc file.
# `acc_flat_reg*` matches only the inferred registers (the combinational
# fp32_add LUTs are named `acc_flat[*]_i_*`, no `_reg`). If a future build
# renames them, `report_timing_summary` will show the violation return and the
# filter needs adjusting.
#
# SAFETY: correct only while the >= 5-cycle spacing invariant holds — enforced
# structurally by the FSM (rowblock_valid fires once per ST_ISSUE, with
# WBITS_BEATS >= 2). Re-derive if the weight-stream FSM ever issues more
# densely (e.g. the deferred ISSUE-bubble removal).
# IS_SEQUENTIAL keeps only the actual FD registers; without it the filter also
# matches the input-cone LUTs (acc_flat_reg[N]_i_M), which aren't valid -to
# endpoints (one warning each).
set_multicycle_path -setup 4 -to [get_cells -hierarchical -filter {IS_SEQUENTIAL && NAME =~ *acc_flat_reg*}]
set_multicycle_path -hold  3 -to [get_cells -hierarchical -filter {IS_SEQUENTIAL && NAME =~ *acc_flat_reg*}]
