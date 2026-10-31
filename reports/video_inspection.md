# Video Inspection — Section 1 Summary

See `visual_inspection/video_inspection.md` for the detailed automated report and preview assets.

## Manually verified from representative frames

- Frame 0 visibly shows the burned-in clock as **2017/12/22 11:12:43**.
- Denser samples suggest the displayed clock reaches approximately **11:14:31 around video time 99 seconds**, then remains unchanged through the sampled tail. This needs OCR validation; do not assume linear timestamp progression across the apparent plateau.
- The final few seconds contain long runs of repeated or near-identical frames.
- The camera is a fixed, elevated view into a retail store.
- Multiple people are already visible at frame zero; the exact baseline must be labelled manually.
- The doorway is not fully visible; the bottom boundary is the candidate entry/exit area.
- The entrance band must include an upper approach margin because first detections may occur slightly above the bottom edge.
- Display stands, shelving and people near the counter produce occlusion and overlap.

## Geometry decision status

**Not finalized.** The bottom-edge samples are supplied for manual review. Do not use the broad preview crop as the final entrance band. We will choose normalized/pixel coordinates only after examining multiple actual movement sequences, including people who enter, exit, pause, and turn around.

## Initial occupancy decision status

**Not finalized.** Count distinct identifiable people visible in frame 0 manually. Do not log them as IN events. This is a camera-visible baseline, not the store-wide population.
