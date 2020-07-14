from collections import defaultdict

import torch

from .base import Tracker
from .tr_mot.multitracker import JDETracker, STrack


class TRMOTTracker(Tracker):

    def __init__(self, type_names, frame_rate,
                 max_age=2, min_iou=0.2,
                 feature_thres=0.7, feature_buffer_size=1):
        super(TRMOTTracker, self).__init__(type_names, frame_rate)
        STrack.reset_id()
        self.trackers = {}
        for obj_type in self.type_names:
            self.trackers[obj_type] = JDETracker(
                int(max_age * frame_rate), feature_thres,
                1 - min_iou, 1 - min_iou / 2)
        self.feature_buffer_size = int(feature_buffer_size * frame_rate)
        self.active_tracks = set()

    def convert_to_tracks(self, detection):
        grouped_tracks = defaultdict(list)
        for obj_i in range(len(detection)):
            obj_type = self.type_names(detection.object_types[obj_i].item())
            bbox = detection.image_boxes[obj_i].numpy()
            tlwh = STrack.tlbr_to_tlwh(bbox)
            score = detection.detection_scores[obj_i].item()
            feature = detection.image_features[obj_i].numpy().copy()
            track = STrack(
                tlwh, score, feature, obj_i, self.feature_buffer_size)
            grouped_tracks[obj_type].append(track)
        return grouped_tracks

    def get_tracked_detection(self, detection):
        track_ids = torch.zeros((len(detection)), dtype=torch.int)
        states = torch.zeros((len(detection)), dtype=torch.int)
        track_boxes = torch.zeros((len(detection), 4))
        image_speeds = torch.zeros((len(detection), 2))
        for tracker in self.trackers.values():
            for track in tracker.tracked_stracks:
                self.active_tracks.add(track.track_id)
                obj_i = track.obj_index
                track_ids[obj_i] = track.track_id
                states[obj_i] = track.state
                track_boxes[obj_i] = torch.as_tensor(
                    track.tlbr, dtype=torch.float)
                speed = torch.as_tensor([
                    track.mean[4], track.mean[5] + track.mean[7] / 2])
                image_speeds[obj_i] = speed * self.frame_rate
        detection.track_ids = track_ids
        detection.track_states = states
        detection.track_boxes = track_boxes
        detection.image_speeds = image_speeds
        ongoing_track_ids = set()
        for tracker in self.trackers.values():
            ongoing_track_ids.update([
                t.track_id for t in tracker.tracked_stracks])
            ongoing_track_ids.update([
                t.track_id for t in tracker.lost_stracks])
        finished_track_ids = self.active_tracks - ongoing_track_ids
        self.active_tracks = self.active_tracks - finished_track_ids
        finished_track_ids = torch.as_tensor(
            [*finished_track_ids], dtype=torch.int)
        return finished_track_ids

    def __call__(self, detection):
        grouped_tracks = self.convert_to_tracks(detection)
        for obj_type, tracker in self.trackers.items():
            tracker.update(grouped_tracks[obj_type])
        finished_track_ids = self.get_tracked_detection(detection)
        return finished_track_ids
