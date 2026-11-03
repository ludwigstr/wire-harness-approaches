# The 36 selected clips (final_v1 tracking)

The 36 clips the user selected on 2 Nov 2026 ("best videos"). All refits from 2 Nov 2026 on use only these clips.
The tracking is final_v1 from 1 Nov 2026, the newest skeletonisation. These files are copies of the matching files in
`skeleton_from_export/results/`, which holds all 107 clips. See that folder's README for how the tracking is made.

| folder | content |
|---|---|
| `npz/` | skeletons: `c0..c3` (T, 16, 2) in metres, node 0 = hub; `t0`/`t1`, `fps`, `source_frames`, `reset` (T, 4), `pink_len_mm` (T) |
| `check_videos/` | camera image with the tracked skeleton drawn on top |
| `splitscreen_videos/` | camera + our skeleton, segmentation, raw skeleton side by side |

The node positions are identical to `newton_fit_t36/data/tracks36/`. Those files only lack the extra fields `reset`
and `pink_len_mm`.

Clips: 21 Oct 160124 162340 162632 163244 163624 164010 164524 164755 165018 165246 165753 170550 171620 171842
172103 172318 172716 172928 173234 173628 173858 174117 174701 175638 180139 180422 180646; 26 Oct 125511 130128
130441; 28 Oct 100701 101001 101510 105803 114103 121810.

The user's list said "174710". That clip does not exist. The only 17:47 clip of 21 Oct is 20261021_174701, used instead.
