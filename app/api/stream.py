from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/stream", tags=["Video Streaming"])


@router.get("/mjpeg")
def stream_mjpeg(request: Request):
    """
    Streams multipart/x-mixed-replace MJPEG feed for browser inspection
    with inference reticles overlaid at 30 FPS.
    """
    stream_mgr = request.app.state.stream_mgr
    return StreamingResponse(
        stream_mgr.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
