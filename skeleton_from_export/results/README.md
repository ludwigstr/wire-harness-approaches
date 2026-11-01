# Skeleton results: 88 clips (78 ok, 10 check)

Produced with `scripts/track_replay_folder.py` from the segmentation exports of 21/23/26/28 Oct 2026, as the
loop of 31 Oct - 1 Nov 2026 converted them (`scripts/skeleton_convert_new.sh` in the main project). Updated as
more clips finish.

- `npz/<clip>.npz`: `c0`..`c3` (T, 16, 2) in metres on the board, node 0 at the hub; `t0`, `t1`, `fps` (4.0 assumed);
  `source_frames` = source frame number of each row. Same format as `data/tracks14`.
- `videos/skeleton_check_<clip>.mp4`: the 16-node skeletons on the camera image (red c0, blue c1, green c2, magenta c3).
- `checks.csv`: the automatic checks as the loop wrote them (clips converted before it existed are only in the table).
- `review_log.txt`: the per-clip review notes.

**status**: the reviewed verdict (the review column overrides the numeric `checks.csv` status). `ok` = the stills and the flagged frames look right; `check` = a tracking defect was found, the
review column says which cable and frames to exclude. Three kinds of defect occur: (1) two cables swap strands in a
loop (160124, 180139, 175358); (2) a cable follows the neighbouring bundle strand or skips its own loop for some
frames (123828, 124048, 103236, 100222, 175638); (3) the recording itself has a cut, i.e. dropped frames, where the
hub and a mover jump ~200 mm between consecutive frames (122521, 122147). Single-frame loops of the green cable at the
hub (< 8 % length) are noted but left as ok.

| clip | frames used | tracked | dropped | cable length mm | spread % per cable | max jump mm per cable | status | review |
|---|---|---|---|---|---|---|---|---|
| 20261021_160124 | 10-175 | 166/166 | [] | 547-962 | 18 9 17 17 | 47 34 47 30 | check | red/green swap strands in the loop at the hub in frames 88-98 (red 617->550 mm); endpoints right. Recommend excluding frames 88-98 |
| 20261021_162025 | 10-174 | 165/165 | [] | 607-914 | 6 3 7 3 | 17 15 26 20 | ok |  |
| 20261021_162340 | 10-184 | 175/175 | [] | 619-941 | 6 6 6 11 | 18 28 23 32 | ok |  |
| 20261021_162632 | 10-180 | 171/171 | [] | 618-918 | 5 4 3 10 | 15 13 15 29 | ok |  |
| 20261021_162931 | 10-178 | 169/169 | [] | 599-944 | 8 8 13 9 | 28 21 21 27 | ok |  |
| 20261021_163244 | 10-132 | 123/123 | [] | 617-941 | 5 5 8 11 | 20 23 27 23 | ok |  |
| 20261021_163624 | 10-179 | 170/170 | [] | 619-915 | 6 2 4 15 | 21 25 19 22 | ok |  |
| 20261021_173234 | 10-163 | 154/154 | [165, 166] | 620-950 | 7 9 7 8 | 22 32 40 31 | ok |  |
| 20261021_173628 | 10-177 | 168/168 | [179, 180] | 613-927 | 7 5 14 6 | 21 21 37 28 | ok |  |
| 20261021_173858 | 10-169 | 160/160 | [171, 172] | 619-915 | 8 3 5 6 | 25 17 35 24 | ok |  |
| 20261021_174117 | 10-183 | 174/174 | [185, 186] | 620-947 | 9 7 6 7 | 25 27 32 45 | ok |  |
| 20261021_174347 | 10-161 | 152/152 | [163, 164] | 612-925 | 10 8 11 9 | 26 29 42 22 | ok |  |
| 20261021_174701 | 10-182 | 173/173 | [184, 185] | 619-946 | 8 7 8 8 | 25 30 25 24 | ok |  |
| 20261021_174921 | 10-178 | 169/169 | [180, 181] | 619-946 | 7 7 4 6 | 25 35 25 34 | ok |  |
| 20261021_175141 | 10-182 | 173/173 | [] | 590-929 | 10 5 11 7 | 28 20 43 36 | ok |  |
| 20261021_175358 | 10-161 | 152/152 | [163, 164] | 479-966 | 24 8 34 14 | 41 30 40 33 | check | check; cable 0 (red) on a shortcut strand (~555 vs 618 mm) from 129 to the end (161), cable 2 (green) on the longer neighbouring strand (~690 vs 638) in 127-147, cable 3's rise 826->909 in 150-161 is smooth (straightening). Recommend cutting at 126 or excluding cables 0 and 2 from 127 |
| 20261021_175638 | 10-164 | 155/155 | [166, 167] | 555-980 | 13 10 19 5 | 32 38 43 36 | check | check (status raised from ok); cable 2 (middle mover) flickers between its own strand (~640 mm) and the neighbouring bundle strand (~570 mm) in frames 90-95 and 136-155, endpoints right. Recommend excluding cable 2 in those frames. 175850 ok |
| 20261021_175850 | 10-161 | 152/152 | [163, 164] | 609-931 | 8 5 13 5 | 35 21 31 42 | ok |  |
| 20261021_180139 | 10-165 | 156/156 | [167] | 537-989 | 24 10 22 8 | 42 43 41 37 | check | check (status raised from ok); cables 0 and 1 swap strands in the loop at the top-left hub: blue too long (985 vs 897 mm) from 121, red too short (545 vs 620) from 124, both to the end (165). Recommend cutting at 120 or excluding cables 0-1 from 121 |
| 20261021_180422 | 10-185 | 176/176 | [] | 619-957 | 7 7 6 13 | 21 31 22 26 | ok | ok; cable 3 length rises smoothly 854->910 mm over frames 155-163 and stays flat to the end (cable straightened by the mover, no flicker or jump), gap-free clip of 176 frames. 180646, 141131 ok |
| 20261021_180646 | 10-156 | 147/147 | [158, 159] | 619-961 | 5 8 8 4 | 35 48 30 39 | ok |  |
| 20261023_141131 | 10-190 | 181/181 | [192, 193] | 618-966 | 4 8 8 4 | 15 22 21 19 | ok |  |
| 20261026_125511 | 10-177 | 168/168 | [179, 180] | 615-932 | 5 5 4 1 | 28 37 55 60 | ok | 60 mm jump of cable 3 at frame 155 is the mover moving (length constant) -> ok. 130441, 130128 ok |
| 20261026_130128 | 10-187 | 178/178 | [189, 190] | 616-948 | 2 7 8 1 | 13 21 23 19 | ok |  |
| 20261026_130441 | 10-173 | 164/164 | [175, 176] | 590-926 | 12 10 5 2 | 20 26 12 27 | ok |  |
| 20261026_130734 | 10-168 | 159/159 | [170, 171] | 589-904 | 9 3 11 3 | 13 13 31 14 | ok | ok; first clip of the 26.10. series, different choreography (hub starts bottom-left, moves to top-left), hub identified correctly. 095144, 095519 ok |
| 20261028_095144 | 10-159 | 150/150 | [161, 162] | 613-903 | 6 3 5 2 | 18 19 37 28 | ok |  |
| 20261028_095519 | 10-165 | 156/156 | [167, 168] | 615-907 | 6 7 15 4 | 17 16 48 24 | ok |  |
| 20261028_095907 | 10-125 | 116/116 | [127, 128] | 616-905 | 9 7 28 2 | 23 29 72 34 | ok | ok with note; green loops in the bundle at the hub in single frames 101, 104, 108 (72 mm), otherwise clean |
| 20261028_100222 | 10-146 | 137/137 | [148, 149] | 444-926 | 6 6 40 2 | 22 25 70 24 | check | check; cable 2 (middle mover) on the bundle strand instead of its own bow in frames 126-133 (444-471 mm vs 640), back from 134; single-frame loops at 28, 37, 122. Recommend excluding cable 2 in 126-133. 100701 ok |
| 20261028_100701 | 10-174 | 165/165 | [176, 177] | 615-932 | 6 7 13 3 | 18 20 44 19 | ok |  |
| 20261028_101001 | 10-170 | 161/161 | [172, 173] | 616-907 | 7 7 21 2 | 23 24 54 33 | ok | ok; green loops in the bundle at the hub in frames 121-127 (54 mm) |
| 20261028_101227 | 10-174 | 165/165 | [176, 177] | 615-945 | 10 9 17 8 | 25 21 47 35 | ok | ok; green loops in the bundle at the hub in frames 125-130 (47 mm). 101510 ok |
| 20261028_101510 | 10-175 | 166/166 | [177, 178] | 610-911 | 17 9 14 3 | 33 22 36 25 | ok |  |
| 20261028_101743 | 10-177 | 168/168 | [179, 180] | 615-936 | 7 7 20 3 | 20 15 59 13 | ok | ok; green loops in the bundle at the hub in single frames 130 and 132 (59 mm) |
| 20261028_102001 | 10-167 | 158/158 | [169, 170] | 615-915 | 6 6 18 8 | 21 18 56 37 | ok | ok; green loops in the bundle at the hub in single frames 38-39 and 126 (56 mm) |
| 20261028_102229 | 10-197 | 188/188 | [] | 607-936 | 12 8 28 3 | 30 24 67 18 | ok | ok with note; green loops in the bundle at the hub in scattered frames 123-134 (67 mm), otherwise clean |
| 20261028_102459 | 10-176 | 167/167 | [178, 179] | 612-934 | 8 8 26 2 | 25 23 67 27 | ok | ok with note; green loops in the bundle at the hub in single frames 127 and 130 (67 mm), otherwise clean |
| 20261028_102724 | 10-164 | 155/155 | [166, 167] | 618-939 | 10 8 19 3 | 21 32 97 73 | ok | ok with note; green loops in the bundle at the hub in frames 118-119 (97 mm), cable 3's 73 mm at 118 is the bottom mover moving (length constant), otherwise clean |
| 20261028_103000 | 10-162 | 153/153 | [164, 165] | 607-938 | 6 7 24 3 | 19 18 57 31 | ok | ok with note; green loops in the bundle at the hub in frames 121, 124-126 (57 mm), otherwise clean |
| 20261028_103236 | 10-173 | 164/164 | [175, 176] | 577-923 | 5 6 15 25 | 18 15 17 84 | check | check; cable 3 (bottom mover) has a slack loop at its mover that the tracer alternately follows (~805 mm) or cuts (~665 mm): flickers in 72-83, 113, 119, 147-150, and stays on the shortcut from 156 to the end (173). Endpoints right, no cable swap; cable 2 ~60 mm short in 162-166. Recommend excluding cable 3 in the flagged frames or cutting at 155. 110851 ok |
| 20261028_103501 | 10-177 | 168/168 | [179, 180] | 614-946 | 7 9 2 2 | 22 17 27 25 | ok |  |
| 20261028_103722 | 10-161 | 152/152 | [163, 164] | 613-926 | 15 5 4 4 | 28 27 24 26 | ok |  |
| 20261028_104336 | 10-179 | 170/170 | [181, 182] | 612-911 | 9 4 23 8 | 25 24 62 39 | ok | ok with note; green loops in the bundle at the hub in frames 125-127 and 140 (62 mm), otherwise clean |
| 20261028_104630 | 10-169 | 160/160 | [171, 172] | 615-956 | 11 10 21 2 | 31 19 61 28 | ok | ok with note; green loops in the bundle at the hub in single frames 117, 119, 122-123, 127 (61 mm), otherwise clean |
| 20261028_104941 | 10-169 | 160/160 | [171, 172] | 618-903 | 15 4 22 8 | 33 21 59 34 | ok | ok with note; green loops in the bundle at the hub in frames 119, 123-124 (59 mm), otherwise clean. 105230 ok |
| 20261028_105230 | 10-170 | 161/161 | [172, 173] | 617-952 | 6 9 7 3 | 26 29 43 29 | ok |  |
| 20261028_105524 | 10-155 | 146/146 | [157, 158] | 609-942 | 10 8 20 2 | 28 25 48 30 | ok | ok with note; green loops in the bundle at the hub in single frames (28, 111, 114, 116, 117), otherwise clean. 110056, 105803 ok |
| 20261028_105803 | 10-155 | 146/146 | [157, 158] | 618-936 | 6 7 3 8 | 28 25 31 33 | ok |  |
| 20261028_110056 | 10-169 | 160/160 | [171, 172] | 618-924 | 6 5 3 2 | 20 18 33 39 | ok |  |
| 20261028_110352 | 10-163 | 154/154 | [165, 166] | 616-919 | 9 5 20 8 | 27 13 61 37 | ok | ok with note; green makes a single-frame loop in the bundle at the hub at frame 121 (61 mm), otherwise clean. 110625, 111116, 111329, 111602, 111837, 112107, 112319, 112535, 112821, 113039 ok |
| 20261028_110625 | 10-169 | 160/160 | [171, 172] | 616-944 | 8 8 6 11 | 26 21 20 38 | ok |  |
| 20261028_110851 | 10-171 | 162/162 | [173, 174] | 615-927 | 10 5 5 2 | 29 18 30 40 | ok |  |
| 20261028_111116 | 10-168 | 159/159 | [170, 171] | 616-925 | 6 6 3 2 | 24 18 27 23 | ok |  |
| 20261028_111329 | 10-174 | 165/165 | [176, 177] | 614-932 | 8 7 6 3 | 23 16 34 29 | ok |  |
| 20261028_111602 | 10-162 | 153/153 | [164, 165] | 616-956 | 8 9 4 2 | 28 21 17 19 | ok |  |
| 20261028_111837 | 10-157 | 148/148 | [159, 160] | 614-954 | 6 9 17 1 | 18 25 45 32 | ok |  |
| 20261028_112107 | 10-154 | 145/145 | [156] | 614-934 | 6 7 4 2 | 37 31 53 48 | ok |  |
| 20261028_112319 | 10-172 | 163/163 | [174, 175] | 615-942 | 6 8 3 2 | 21 15 15 20 | ok |  |
| 20261028_112535 | 10-161 | 152/152 | [163, 164] | 615-925 | 6 6 3 2 | 20 16 21 22 | ok |  |
| 20261028_112821 | 10-165 | 156/156 | [167, 168] | 614-923 | 8 6 4 2 | 26 21 34 31 | ok |  |
| 20261028_113039 | 10-164 | 155/155 | [166, 167] | 616-951 | 5 8 5 1 | 25 20 31 42 | ok |  |
| 20261028_113317 | 10-156 | 147/147 | [158, 159] | 618-939 | 6 8 16 1 | 22 32 60 56 | ok | ok with note; green makes a single-frame loop in the bundle at the hub at frame 117 (60 mm), otherwise clean |
| 20261028_113545 | 10-153 | 144/144 | [155, 156] | 612-915 | 7 5 26 2 | 53 23 61 43 | ok | ok with note; green on the neighbouring strand of the bundle in single frames 122 and 125 (61 mm jump), otherwise clean. 113827 ok |
| 20261028_113827 | 10-159 | 150/150 | [161, 162] | 617-944 | 6 7 13 8 | 22 26 43 40 | ok |  |
| 20261028_114103 | 10-155 | 146/146 | [157, 158] | 491-923 | 13 5 24 4 | 33 18 48 40 | ok | ok; red/green/blue share a tight loop above the middle mover, green on the neighbouring strand for single frames 105 and 109; red ~140 mm short in the last two frames 154-155. 114335 ok |
| 20261028_114335 | 10-152 | 143/143 | [154, 155] | 615-933 | 6 7 12 10 | 38 15 37 39 | ok |  |
| 20261028_114911 | 10-154 | 145/145 | [156, 157] | 616-901 | 6 2 2 2 | 36 29 47 67 | ok | 67 mm jump of cable 3 at frame 118 is a real fast mover move (all lengths constant) -> ok. 120651, 120430, 115924, 115704, 115424, 115140 ok |
| 20261028_115140 | 10-142 | 133/133 | [144, 145] | 617-956 | 5 10 3 2 | 27 29 43 42 | ok |  |
| 20261028_115424 | 10-175 | 166/166 | [177, 178] | 616-933 | 6 7 16 3 | 21 18 59 40 | ok |  |
| 20261028_115704 | 10-158 | 149/149 | [160, 161] | 610-909 | 11 4 11 2 | 26 29 39 59 | ok |  |
| 20261028_115924 | 10-159 | 150/150 | [161, 162] | 616-916 | 7 5 2 8 | 27 22 38 42 | ok |  |
| 20261028_120200 | 10-171 | 162/162 | [173, 174] | 616-924 | 6 6 17 10 | 24 21 54 35 | ok | ok; green loops in the bundle at the hub in single frames 121 and 127 (54 mm). 103722, 103501 ok |
| 20261028_120430 | 10-172 | 163/163 | [174, 175] | 616-949 | 5 8 12 2 | 19 20 38 26 | ok |  |
| 20261028_120651 | 10-173 | 164/164 | [175, 176] | 616-928 | 6 7 6 8 | 31 19 26 35 | ok |  |
| 20261028_120911 | 10-158 | 149/149 | [160, 161] | 617-913 | 11 9 4 3 | 37 33 30 31 | ok |  |
| 20261028_121810 | 10-164 | 155/155 | [166, 167] | 549-931 | 6 7 21 2 | 19 20 33 23 | ok | ok; green and blue run bundled, green briefly on the neighbouring strand in 85-88 and 145-164 (<=8% length), no jumps |
| 20261028_122147 | 10-129 | 120/120 | [131, 132] | 530-929 | 22 9 44 4 | 147 129 150 233 | check | same recording discontinuity as 122521, between frames 118 and 119 (hub 739->364 px, bottom mover jumps, raw change 2x normal); cable 2 on the wrong strand in 119-123, back from 124. Recommend cutting at 118 or excluding cable 2 in 119-123. 122819 ok |
| 20261028_122521 | 10-126 | 117/117 | [128, 129] | 540-910 | 20 10 43 8 | 122 101 108 202 | check | recording discontinuity between frames 111 and 112 (hub and bottom mover jump ~200 mm in the class map itself, raw-image change 2x normal) -> dropped frames in the RECORDING; cable 2 traces the wrong strand (magenta's) in frames 112-123, back on track from 124. Recommend cutting at 111 or excluding cable 2 in 112-123. 123311, 123040, 122819, 123540 ok |
| 20261028_122819 | 10-158 | 149/149 | [160, 161] | 615-970 | 6 11 10 1 | 22 26 35 42 | ok |  |
| 20261028_123040 | 10-154 | 145/145 | [156, 157] | 616-902 | 6 7 13 2 | 19 19 42 45 | ok |  |
| 20261028_123311 | 10-165 | 156/156 | [167, 168] | 617-948 | 6 8 15 5 | 20 14 47 22 | ok |  |
| 20261028_123540 | 10-155 | 146/146 | [157, 158] | 616-906 | 15 8 7 9 | 42 24 31 57 | ok |  |
| 20261028_123828 | 10-160 | 151/151 | [162, 163] | 534-1005 | 19 14 5 6 | 20 52 17 29 | check | cable 0 ~55 mm short in frames 138-160 -- strand reaches the top image edge under the gantry (possible occlusion), keep as check; 124048: cable 2 ~80 mm short in frames 64-69, 75, 122-130 -- shorter loop below the middle mover, keep as check |
| 20261028_124048 | 10-169 | 160/160 | [] | 534-906 | 7 3 30 5 | 22 27 48 50 | check |  |
| 20261028_124319 | 10-199 | 190/190 | [] | 615-953 | 12 8 5 1 | 38 31 41 66 | ok | 66 mm jump of cable 3 at frame 140 is a real fast mover move (length constant 804 mm, strand follows) -> ok; 124555 ok |
| 20261028_124555 | 10-156 | 147/147 | [158, 159] | 615-910 | 6 8 5 6 | 15 31 34 30 | ok |  |
| 20261028_124829 | 10-162 | 153/153 | [] | 619-923 | 14 10 14 11 | 43 41 46 42 | ok |  |
