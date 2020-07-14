__author__ = 'Lijun Yu'


def get_tracker(name):
    if name == 'Towards-Realtime-MOT':
        from .trmot import TRMOTTracker
        return TRMOTTracker
    else:
        raise NotImplementedError('Tracker<%s> not found' % (name))
