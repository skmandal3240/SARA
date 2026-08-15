# see

when: a camera frame arrives on a SARA Cam / drone / phone
need_grant: camera
do_not: upload raw frames; call cloud; share_raw without preview

## steps

1. Preview+approve `camera`. Cloud stays denied unless a separate grant exists.
2. Run vision encoder on-device (INT8 if the profile says so).
3. Caption / alert locally. Mesh may move **embeddings**, not the JPEG, unless `share_raw`.
4. Audit the inference (model hash + quant + adapter id).
