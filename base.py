from enum import EnumMeta

import torch


class Tracker(object):

    def __init__(self, type_names: EnumMeta, frame_rate: float):
        self.type_names = type_names
        self.frame_rate = frame_rate

    def __call__(self, detection) -> torch.Tensor:
        '''
        detection: detectors.base.Detection from 
            https://github.com/CMU-INF-DIVA/detectors
        Track attributes added: track_ids, track_states, track_boxes, 
            image_speeds
        returns finished_track_ids
        '''
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.video_name)
