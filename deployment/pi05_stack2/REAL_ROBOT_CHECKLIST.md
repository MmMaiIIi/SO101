# Real robot operating notes

The Stack2 v1 package has already executed real motor commands and demonstrated one visually successful blue-on-yellow task. That does not certify a different arm, calibration, camera layout, or Stack3 scene.

Before using another setup, supply its own local device IDs, calibration, camera mapping, and safety profile; run the read-only preflight commands in [README.md](README.md). Never commit these private hardware files. Keep the saved absolute position/calibration gate enabled. A task result and runtime termination must be reported separately.

The public v1 result had 1,187 policy commands and ended in a post-task `SAFETY_ABORT`: a shoulder-lift retreat target exceeded the saved lower bound by 0.14135° and was not sent. Do not widen limits to reproduce the visual task result.
