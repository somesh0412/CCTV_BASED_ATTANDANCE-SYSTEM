import argparse
import getpass
from urllib.parse import urlsplit

from onvif import ONVIFCamera


def parse_args():
    parser = argparse.ArgumentParser(
        description="List ONVIF media profiles and sanitized RTSP stream details."
    )
    parser.add_argument("--host", required=True, help="Camera ONVIF host or IP address")
    parser.add_argument(
        "--port",
        required=True,
        type=int,
        help="Camera ONVIF service port",
    )
    return parser.parse_args()


def sanitize_uri(uri):
    """Remove credentials from a URI before displaying it."""
    if not uri:
        return "Unavailable"

    try:
        parsed = urlsplit(str(uri))
        if not parsed.scheme or not parsed.netloc:
            return "<redacted URI>"

        hostname = parsed.hostname or "<redacted-host>"
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"

        return f"{parsed.scheme}://{host}{parsed.path}"
    except (TypeError, ValueError):
        return "<redacted URI>"


def format_value(value):
    return str(value) if value is not None else "Unavailable"


def profile_details(media, profile):
    request = media.create_type("GetStreamUri")
    request.ProfileToken = profile.token
    request.StreamSetup = {
        "Stream": "RTP-Unicast",
        "Transport": {"Protocol": "RTSP"},
    }
    stream_uri = media.GetStreamUri(request).Uri

    encoder = getattr(profile, "VideoEncoderConfiguration", None)
    if encoder is None:
        return {
            "resolution": "Unavailable (no video encoder configuration)",
            "encoding": "Unavailable",
            "fps": "Unavailable",
            "bitrate": "Unavailable",
            "uri": sanitize_uri(stream_uri),
        }

    resolution = getattr(encoder, "Resolution", None)
    if resolution is None:
        resolution_text = "Unavailable"
    else:
        width = getattr(resolution, "Width", None)
        height = getattr(resolution, "Height", None)
        if width is None or height is None:
            resolution_text = "Unavailable"
        else:
            resolution_text = f"{width} x {height}"

    rate_control = getattr(encoder, "RateControl", None)
    fps = getattr(rate_control, "FrameRateLimit", None)
    bitrate = getattr(rate_control, "BitrateLimit", None)

    return {
        "resolution": resolution_text,
        "encoding": format_value(getattr(encoder, "Encoding", None)),
        "fps": format_value(fps),
        "bitrate": format_value(bitrate),
        "uri": sanitize_uri(stream_uri),
    }


def main():
    args = parse_args()
    username = input("ONVIF username: ")
    password = getpass.getpass("ONVIF password: ")

    try:
        camera = ONVIFCamera(
            args.host,
            args.port,
            username,
            password,
        )
        media = camera.create_media_service()
        profiles = media.GetProfiles()
    except Exception as error:
        print("Could not connect to the camera or retrieve ONVIF profiles.")
        return 1

    if not profiles:
        print("The camera returned no ONVIF media profiles.")
        return 0

    for index, profile in enumerate(profiles, start=1):
        print(f"\nProfile {index}")
        print(f"Name: {format_value(getattr(profile, 'Name', None))}")
        print(f"Token: {format_value(getattr(profile, 'token', None))}")

        try:
            details = profile_details(media, profile)
        except Exception:
            print("Video configuration: Unavailable")
            print("Encoding: Unavailable")
            print("FPS: Unavailable")
            print("Bitrate: Unavailable")
            print("RTSP URI: Unavailable")
            continue

        print(f"Resolution: {details['resolution']}")
        print(f"Encoding: {details['encoding']}")
        print(f"FPS: {details['fps']}")
        print(f"Bitrate: {details['bitrate']}")
        print(f"RTSP URI: {details['uri']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
