# Cable skeletons from a segmentation export folder

`scripts/track_replay_folder.py` turns the export folder of the segmentation pipeline (for example
`20261028_124829/`) into cable skeletons for testing algorithms: 4 cables x 16 nodes per frame, in metres on
the board. The output has the same format as `data/tracks14`, so every evaluation script in this repo that
reads tracks (`--tracks`) can use it directly.

## Input

The export folder as the pipeline writes it. The script reads only, for each processed frame:

- `visualization/panels/<k>_frame_<n>/01_raw.png`: the camera image (2000 x 2660)
- `visualization/panels/<k>_frame_<n>/02_class_map.png`: the class map (yellow = cable, green = mover)

Here `<n>` is the source frame number. No source video is needed, and the export's own MP4s are not used.

## What it does

1. **Choose the frames.** It takes the longest run of consecutive source frame numbers and drops duplicates,
   so every step is exactly one source frame.
2. **Rebuild the tracker's inputs.** It writes:
   - the raw images as a source video at 0.8 scale (1600 x 2128, the original camera resolution);
   - a side-by-side video (raw | class map) in a private segmentation folder, set via `WIRE_SEG_DIR`.
3. **Track.** It runs the project's tracking pipeline unchanged, the same one that produced
   `data/tracks14`:
   - `find_board` gives the pixel-to-millimetre transform from the board bezel;
   - `track_harness` finds the five modules per frame from the class map, with stable identities, and the
     hub first;
   - `HarnessTracker` traces each cable from the hub to its mover through the cable pixels and resamples it
     to 16 nodes.
4. **Write the outputs** into `--out`:
   - `<clip>.npz`, with these keys:
     - `c0`..`c3`: arrays of shape (T, 16, 2) in metres; node 0 is at the hub
     - `t0`, `t1`, `fps`
     - `source_frames`: the source frame number of each row
   - `skeleton_check_<clip>.mp4`: the 16-node skeletons drawn on the camera images, plus one still frame.
   - `<clip>.mp4` and `seg_input/`: the rebuilt inputs. They are intermediate files and can be deleted.

## Usage

Run from inside `skeleton_from_export/`. Relative paths resolve against this folder; absolute paths work too.

    python scripts/track_replay_folder.py --folder /path/to/20261028_124829 --out /path/to/20261028_124829/skeleton

Options:
- `--fps` (default 4.0) is written to the npz. The export does not record the frame rate.
- `--out` must not exist yet; the script refuses to overwrite.

Requirements: numpy, scipy, opencv-python, scikit-image, imageio with ffmpeg. `wire_model/perception/` is a
copy of the main project's perception package (2026-10-29).

## Example

`example/` holds the result for the export `20261028_124829`:

- 155 distinct frames; the script used source frames 10..162 (153 frames). Frame 162 appears twice and 163 is
  missing, so 164 and 165 are left out.
- All 153 frames are tracked.
- Cable lengths are 620-920 mm, stable over the recording.
- The largest step is 46 mm per node, around frame 105, where the movers really do move fast.
- `example/skeleton_check_20261028_124829_frame076.png` shows one check frame.

Runtime is about 13 min for 153 frames on CPU.
