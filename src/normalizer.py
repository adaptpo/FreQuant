import numpy as np
from typing import Optional


def normalizer(features: np.array, norm_type_: str, dates: Optional[np.array]):
    """
    Note that features should always have following dimension format and temporal dimension ordering
        i.e. features.shape = (Time, Asset, Feature) & Time is in ascending order (T, N, F)

    """
    if norm_type_ == 'ir':  # Increase Ratio (Rate of Increase vs. Previous Date Prices)
        features[1:] = (features[1:] - features[:-1]) / features[:-1]
        features = features[1:]
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    elif norm_type_ == 'irvpcp':  # Increase Ratio versus Previous Date Closing Price
        p_c_p = np.expand_dims(features[:-1, :, 3], axis=2)  # Previous Closing Price (T, N, 1)
        features[1:] = (features[1:] - p_c_p) / p_c_p
        features = features[1:]
        np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    else:
        raise ValueError(f"Not existing normalization. \'{norm_type_}\'")

    if dates is None:
        pass
    else:
        dates = dates[1:]

    return features, dates
