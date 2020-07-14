from enum import EnumMeta
from typing import Any, Tuple

import torch


class Tracker(object):

    def __init__(self, type_names: EnumMeta, video_name: str,
                 frame_rate: float):
        self.type_names = type_names
        self.video_name = video_name
        self.frame_rate = frame_rate

    def __call__(self, detection: Any) -> Tuple[Any, torch.Tensor]:
        '''
        detection: detectors.base.Detection from 
        https://github.com/CMU-INF-DIVA/detectors
        returns (detection, finished_track_ids)
        '''
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.video_name)
