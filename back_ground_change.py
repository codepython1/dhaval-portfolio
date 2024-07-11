from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import WebRtcMode
import av
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

def callback(frame:av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format = "bgr24")
    
    img = segmentor.removeBG(img,cutThreshold=0.8)

    return av.VideoFrame.from_ndarray(img,format="bgr24")

webrtc_streamer(key="hand-detection",video_frame_callback=callback,media_stream_constraints={"video":True,"audio":False},async_processing=True)



