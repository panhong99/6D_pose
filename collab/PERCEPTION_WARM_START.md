# Warm camera perception for plate demos

The service keeps the fixed and wrist D455 streams, plate YOLOE detector, and
FoundationPose model loaded between pipeline invocations. It sends the same
plate poses to UDP 5005 and updates the same `/dev/shm/*_latest.npz` frames.
It never connects to or moves the robot.

```bash
cd /home/pan/pan/6D_pose
conda run -n foundationpose python calibration/perception_service.py start --plate_color white
conda run -n foundationpose python calibration/perception_service.py status
```

After `start` reports `Warm`, run the usual `bash calibration/run.sh pipeline ...`
command. The plate pipeline automatically reuses a matching live service and
leaves it running on exit. The service log is
`/tmp/6d_pose_perception_service.log`.

Before changing from white to red or taking exclusive camera control:

```bash
conda run -n foundationpose python calibration/perception_service.py stop
conda run -n foundationpose python calibration/perception_service.py start --plate_color red
```

The pipeline refuses a live service with the wrong plate prompt, mesh, camera,
or confidence. A stale/dead service is ignored. The separate basket YOLOE
model in `basket_vision.py` still loads for each markerless basket measurement;
this service currently removes only the repeated plate detector/FoundationPose
and camera startup.
