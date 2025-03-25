import pandas as pd


def format_timedelta(timedelta: pd.Timedelta) -> str:
    components = {
        'days': 'j', 'hours': 'h', 'minutes': 'min',
        'seconds': 's', 'milliseconds': 'ms',
        'microseconds': 'us', 'nanoseconds': 'ns'
    }

    return "".join(
        f"{getattr(timedelta.components, c)}{components[c]}"
        for c in components.keys()
        if getattr(timedelta.components, c) != 0
    )
