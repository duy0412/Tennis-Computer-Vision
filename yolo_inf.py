from utils import read_video, save_video
from trackers import PlayerTracker
from court_line_detector import CourtLineDetector

video_frames = read_video('input_videos/video_input5.mp4')

player_tracker = PlayerTracker(model_path='yolo11n.pt')

player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=False,
                                                     stub_path="tracker_stubs/player_detections_video_input5.pkl"
                                                     )
# Court Line Detector model
court_model_path = "models/keypoints_model.pth"
court_line_detector = CourtLineDetector(court_model_path)
court_keypoints = court_line_detector.predict(video_frames[0])

# choose players
player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
output_video= player_tracker.draw_bboxes(video_frames, player_detections)
save_video(output_video, 'output_videos/video_output3_keypoint_only.avi')