# How-to for a Claude instance: cable skeletons for a drive full of tracking exports

You have a drive with many export folders from the cable segmentation pipeline. Your job is to produce,
for every folder, the cable skeletons (4 cables x 16 nodes per frame, in metres on the board) and a check
video, using `skeleton_from_export/scripts/track_replay_folder.py` from this repository. Read
`skeleton_from_export/README.md` once for what the script does. This file covers how to run it at scale
and how to check the result.

## Rules

- **Never write to, rename or delete anything on the data drive.** Treat it as read-only. All output goes
  to a separate output folder (below).
- **Never overwrite an output.** The script refuses an `--out` that exists; do not work around that by
  deleting it. If a folder must be redone, use a new output name such as `<name>_v2` and say why.
- **Do not change the tracking code** (`wire_model/perception/`) to make a folder pass. If a folder fails,
  record it and report it. These skeletons are used to test algorithms, so a wrong skeleton is worse than
  a missing one.
- **Do not push outputs to GitHub.** They are large. Ask the user where results should go.
- Ask the user before any long or parallel run that would take over the machine.

## 1. Get the code

The repository is private (`ludwigstr/wire-harness-approaches`). If `git clone` asks for credentials, stop
and ask the user to authenticate (for example with `gh auth login`). Never ask for a token in the chat,
and never write a token into a file or a command line.

    git clone https://github.com/ludwigstr/wire-harness-approaches.git
    cd wire-harness-approaches/skeleton_from_export
    python3 -m venv .venv && . .venv/bin/activate
    pip install numpy scipy opencv-python scikit-image "imageio[ffmpeg]"

The script was run with Python 3.11, OpenCV 5.0, imageio 2.37, scikit-image 0.25 and SciPy 1.16. Run
every command from inside `skeleton_from_export/`, because the script imports `wire_model/` from there.

No GPU is needed.

## 2. Find the export folders

An export folder can be used if it contains, for each processed frame:

    <folder>/visualization/panels/<k>_frame_<n>/01_raw.png        camera image, 2000 x 2660
    <folder>/visualization/panels/<k>_frame_<n>/02_class_map.png  yellow = cable, green = mover

The folder name is the recording name, for example `20261028_124829`, and becomes the clip name.

List the candidates without touching anything:

    find /media/<drive> -type d -path "*/visualization/panels" 2>/dev/null | sed 's#/visualization/panels##' > folders.txt
    wc -l folders.txt

Then check each candidate: count its panel subfolders and the ones that have both PNGs. A folder whose
panels lack `02_class_map.png`, or that has only MP4s, cannot be processed by this script. List those
separately for the user and do not try to rebuild them from the MP4s. The MP4s in these exports are often
truncated (`moov atom not found`).

Report to the user before running: how many folders there are, how many are usable, and the total number
of frames.

## 3. Run one folder first

Pick one small folder and run:

    python scripts/track_replay_folder.py --folder /media/<drive>/<name> --out /path/to/skeletons/<name>

Use absolute paths for both. Expect about 5 s per frame on the CPU; 153 frames took about 13 min. Watch
for these lines:

    <name>: 155 distinct frames, using 10..162 (153); dropped [164, 165]
    tracked 153 of 153 frames
    -> .../<name>.npz: c0..c3 (153, 16, 2)

Check the result (section 5) before you start the batch.

## 4. Batch run

Write a small driver script (bash or Python) that, for each usable folder:

- skips the folder if its output folder already exists, so the batch can be restarted;
- runs the command above with `--out /path/to/skeletons/<name>` and writes the log to
  `/path/to/skeletons/logs/<name>.log`;
- after a successful run, deletes only the two intermediate products **in the output folder**:
  `<name>.mp4` (about 330 MB) and `seg_input/` (about 570 MB). Both can be rebuilt from the drive. Keep
  `<name>.npz`, `skeleton_check_<name>.mp4` and the `skeleton_check_*_frame*.png` still frame;
- appends one row to `/path/to/skeletons/summary.csv` with these columns:
  - name, number of distinct frames, frames used, frames tracked
  - dropped frame numbers
  - the per-cable checks from section 5
  - status (`ok` / `check` / `failed`) and the error message

Plan the disk space: each folder needs about 1 GB while it runs and about 10 MB afterwards.

The script is single-threaded for the most part. Run several folders in parallel with
`OMP_NUM_THREADS=2`, but no more processes than about half the CPU cores, and ask the user first how much
of the machine you may use.

Start it detached so it survives the session:

    setsid nohup bash run_all.sh > run_all.log 2>&1 < /dev/null & disown

## 5. Check every result

The script's success message does not mean the skeleton is right. For each `.npz`, compute and record:

| Check | How | Flag when |
|---|---|---|
| Coverage | len(c0) vs. frames used | any frame not tracked |
| Cable length | arc length of each cable per frame, in mm | spread (max - min) > 25 % of its median, or any cable < 200 mm (the reference folder shows 10-14 %) |
| Jumps | mean node displacement between consecutive frames, in mm | > 60 mm per node (the reference folder peaks at 46 mm, a real fast move) |
| Finite | `np.isfinite` | any NaN |

Then look at the still frame of every folder, and at the frames around every flagged jump in the check
video. Node 0 of all four cables is the same hub point by construction, so whether the right mover is the hub can only
be checked by eye: it is the mover all four cables leave from. A cable that leaves its own strand for a
neighbouring one, or a hub on the wrong mover, is a tracking failure: mark it `failed` with the frame numbers. A large jump is fine if the movers really moved
that far between the two frames.

Known behaviour:

- **Frame numbering.** The script uses the longest run of consecutive source frames. Exports often repeat
  one frame and skip the next, for example 162 twice and no 163. Everything after such a gap is left out,
  and the log line says which frames. If a lot is dropped (more than about 10 % of the frames), report it.
- **Frame rate.** The script writes `fps=4.0`; the export does not store the real rate. If the user knows
  the true rate, rerun with `--fps` into a new output folder.
- **Overlay text.** The burned-in text at the top left of `01_raw.png` did not disturb the tracking on the
  test folder.
- **Five modules required.** The setup must have one hub and four movers, each with one cable to the hub.
  If `track_harness found no modules` appears, or fewer than five green blobs are visible in the class
  map, mark the folder `failed`. It is a different setup.

## 6. Report

When done, give the user:

- the counts: processed, `ok`, `check`, `failed`, not usable;
- the path to `summary.csv` and to the output folder;
- every `check` and `failed` folder with the frame numbers and one sentence on what is wrong.

## Output format, for whoever uses the results

`<name>.npz` contains:

- `c0`..`c3`: arrays of shape (T, 16, 2), float, in metres in the board frame: origin at the bottom-left board corner, x to the
  right, y up the board. Node 0 sits at the hub and node 15 at that cable's mover.
- `t0`, `t1`: the first and last used index into the rebuilt video.
- `fps`: the frame rate written by the script (see Known behaviour).
- `source_frames`: the source frame number of each of the T rows.

This is the format of `data/tracks14` in the repository, so any script that takes `--tracks <dir>` can read
a folder of these files.

A reference result for checking your setup is in `example/` (`20261028_124829`). If you have that export
folder, rerunning it must reproduce `example/20261028_124829.npz`, up to small differences from video
encoding.
