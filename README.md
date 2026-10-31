# Store People Counter — Section 1 Package

This package contains the project scaffold and completed **Section 1: Video Inspection** utilities and outputs. The assignment is scoped to offline analysis of the supplied CCTV MP4. It does not implement real-time CCTV monitoring.

## Included

- `input/store_cctv.mp4` — supplied source video, kept unchanged.
- `src/video_inspector.py` — reusable inspection utility.
- `scripts/inspect_video.py` — command-line entry point.
- `reports/visual_inspection/` — metadata report, first frame, representative contact sheets and JSON report.
- `annotations/*template*` — empty annotation templates for later sections.
- `config/` — provisional settings; entrance coordinates are intentionally not calibrated yet.
- `requirements.txt` — dependencies planned for the complete offline pipeline.

## Project location (Windows)

The recommended project directory for this setup is:

```text
F:\Store_People_Counter
```

After extracting the ZIP, confirm that this file exists:

```text
F:\Store_People_Counter\input\store_cctv.mp4
```

## Python environment setup (PowerShell)

Use Python 3.12.x. Open PowerShell and run:

```powershell
cd F:\Store_People_Counter
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## PyTorch and NVIDIA GPU note

Ultralytics depends on PyTorch. The requirements file installs the compatible PyTorch dependency selected by pip, but that does not guarantee that the installed build can use the NVIDIA GPU. First verify the environment:

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, use the official PyTorch installation selector at https://pytorch.org/get-started/locally/ to choose the Windows, pip, Python and CUDA build supported by your driver. Install the selected PyTorch/torchvision command inside the activated `.venv`, then rerun the verification command. The project remains runnable on CPU, although inference will be slower. Do not install a random CUDA wheel or CUDA toolkit version by guesswork.

## Tesseract OCR note

`pytesseract` is a Python wrapper. Actual OCR also requires the Tesseract OCR application to be installed separately on Windows and available on `PATH`. OCR recognition is not part of Section 1; this dependency is included for the planned timestamp stage.

## Run Section 1 inspection

```powershell
python scripts\inspect_video.py --video input\store_cctv.mp4 --project-root .
```

Optional sample times (seconds):

```powershell
python scripts\inspect_video.py --video input\store_cctv.mp4 --project-root . --samples 0,5,10,20,30,40,50,60,70,80,90,100,109
```

The script writes outputs to `cache\frames\` and `reports\visual_inspection\`.

## Verified source-video metadata

- MP4 container; H.264 video stream
- 1270 × 720 pixels
- Average stream frame rate: approximately 13.093 FPS
- 1,452 decoded video frames
- Duration by frame count / average FPS: approximately 110.899 seconds
- Frame-zero clock visibly reads `2017/12/22 11:12:43`
- Sampled clock crops suggest the displayed clock reaches approximately `11:14:31` around video time 99 seconds and then stops changing in the sampled tail; OCR validation is pending.
- The final few seconds contain long runs of repeated or near-identical frames.

The clock must not be extrapolated linearly through the apparent plateau. The MP4 container duration reported by ffprobe is approximately 111.068 seconds. Frame-index / stream-FPS is the processing timeline.

## Section 1 boundaries

This stage inspects the video and prepares reference images. It does **not** run OCR recognition, person detection, tracking, entry/exit counting or headcount calculation. The broad clock and bottom-edge preview crops are not final calibrated regions.

## Next approval gate

Review these outputs before Section 1 is approved:

1. `reports/visual_inspection/full_frame_contact_sheet.jpg`
2. `reports/visual_inspection/clock_region_samples.jpg`
3. `reports/visual_inspection/bottom_edge_samples.jpg`
4. `reports/visual_inspection/video_inspection.md`

Then we will manually establish the frame-zero visible baseline and calibrate the entrance band and upper approach margin from actual movement sequences.
