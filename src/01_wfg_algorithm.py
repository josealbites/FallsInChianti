import pandas as pd
from typing import Dict, Optional

def assess_fall_risk_wfg(
        fall_history_12m: bool,
        feels_unsteady: Optional[bool] = None,
        worries_about_falling: Optional[bool] = None,
        gait_speed_mps: Optional[float] = None,
        tug_time_sec: Optional[float] = None,
        had_fall_injury: Optional[bool] = None,
        had_multiple_falls_12m: Optional[bool] = None,
        is_frail: Optional[bool] = None,
        found_on_floor: Optional[bool] = None,
        had_loc_syncope: Optional[bool] = None,
) -> Dict[str, str]:
    """ Applies the 2022 World Fall Guidelines (WFG) risk stratification algorithm. """
