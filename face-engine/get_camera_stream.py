from onvif import ONVIFCamera

CAMERA_IP = "192.168.145.166"
ONVIF_PORT = 5000

USERNAME = input("Camera username: ")
PASSWORD = input("Camera password: ")

try:
    camera = ONVIFCamera(
        CAMERA_IP,
        ONVIF_PORT,
        USERNAME,
        PASSWORD
    )

    media = camera.create_media_service()

    profiles = media.GetProfiles()

    for profile in profiles:
        print("\nProfile:", profile.Name)
        print("Token:", profile.token)

        request = media.create_type("GetStreamUri")
        request.ProfileToken = profile.token
        request.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"}
        }

        result = media.GetStreamUri(request)
        print("RTSP URL:", result.Uri)

except Exception as error:
    print("Error:", error)