# Section 1 — Video Inspection Report

## Source metadata

| Property | Value |
|---|---|
| Source file | `store_cctv.mp4` |
| Container | MP4 |
| Video codec | H.264 |
| Resolution | 1270 × 720 |
| Average frame rate reported by OpenCV | 13.093000 FPS |
| OpenCV estimated frame count | 1,454 |
| Frames actually decoded | 1,452 |
| Duration derived from decoded frames / FPS | 110.899 seconds |
| File size | 27.67 MB |
| SHA-256 | `f6d18b9025c952f676eda294fb595fae7d68aa32778bd4af386e13355cc572a0` |

> The supplied stream metadata reports approximately 110.899 seconds from frame count / average FPS. Container duration reported separately by ffprobe is approximately 111.068 seconds. Frame index divided by the stream FPS will be the processing timeline used by the application.

## Visual observations

1. The camera is fixed and looks down into the store from an elevated angle.
2. The doorway itself is not fully visible; people appear/disappear through the lower image boundary.
3. The first frame already contains people in the monitored view. These people must be represented by a manually verified initial visible baseline, not synthetic IN events.
4. Display stands, shelving and the checkout area create frequent partial occlusion and overlapping people.
5. The burned-in clock is visible in the top-left region. Frame zero visibly reads **2017/12/22 11:12:43**. This is a visual reading to validate with OCR before final timestamp synchronization.
6. Denser clock samples near the end visually suggest the displayed time reaches approximately **11:14:31 around video time 99 seconds**, then remains unchanged through the sampled tail. Do not assume a simple linear mapping from video elapsed time to CCTV clock time until OCR validation is completed.
7. The final few seconds contain long runs of repeated or near-identical decoded frames. The tail should be inspected before interpreting any events there.
8. The lower portion of the frame is the entrance-analysis candidate area. The final band and its upper approach margin are deliberately not frozen in this inspection stage.

## Generated inspection assets

- `frame_zero.jpg` — original-size first frame.
- `full_frame_contact_sheet.jpg` — representative full-frame samples.
- `clock_region_samples.jpg` — top-left clock crop samples.
- `bottom_edge_samples.jpg` — broad lower-frame samples for entrance geometry review.
- `bottom_edge_sequence_01.jpg` onward — one-second bottom-edge sequence pages for movement review.
- `clock_samples_90_to_end.jpg` — denser clock samples near the end of the recording.
- `inspection_report.json` — machine-readable metadata and sample frame index mapping.

## Section 1 limitations / pending decisions

- Exact entrance band and approach-margin coordinates are not yet calibrated.
- Initial visible occupancy and staff/customer/unknown roles are not yet manually counted.
- OCR accuracy and clock continuity are not yet formally validated; the apparent clock plateau near the end needs confirmation.
- The apparent repeated-frame tail needs confirmation and should not be treated as extra movement.
- No detection, tracking, entry/exit or occupancy algorithm has been run in this section.

## Section 1 acceptance checks

- [x] MP4 can be opened and decoded.
- [x] Width, height, FPS and frame count recorded.
- [x] Representative frames extracted.
- [x] Clock preview crops generated.
- [x] Bottom-edge preview crops generated.
- [ ] Entrance band and upper approach margin manually calibrated (next inspection/annotation decision).
- [ ] Frame-zero baseline manually annotated (ground-truth section).
- [ ] Clock OCR continuity validated (timestamp section).
