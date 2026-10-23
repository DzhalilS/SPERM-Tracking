# SPERM-Tracking

Open-code pipeline for classifying human sperm motility from ordinary microscope videos, without a commercial CASA system.

Tracks of individual spermatozoa (TrackMate in Fiji) are cleaned in Python, described by a few kinematic features and clustered with HDBSCAN into the WHO motility categories: **RP** (rapidly progressive), **SP** (slowly progressive), **NP** (non-progressive) and **IM** (immotile). UMAP is used only to draw the result in 2D.

First-author poster: Salimov D., Mamaev T., Smigulina L., Kireev I., Sergeyev O. *Clusterization of Sperm Motility Based on Video Tracking Data Analysis.* mCARM 2026, St Petersburg, May 2026.

## Why

Manual assessment depends on a trained technician, who has to remember which cells were already counted, and the result is a subjective percentage. CASA systems give objective kinematics, but they are expensive. The goal here is a cheap, reproducible alternative built from free tools (Fiji/TrackMate and Python) that gives every cell a numerical description and a transparent category.

## Pipeline

```
video  ->  Fiji / TrackMate  ->  spots + tracks tables (.xlsx)
                                         |
                    1. filtering   (FILTRATION OF TRACKMATE TRACKS.py)
                                         |
              +--------------------------+--------------------------+
              |                                                     |
 2a. threshold baseline                                2b. HDBSCAN clustering
 (MANUAL CLUSTERIZATION MAIN.py)                       (SPERM CLUSTERIZATION NO CASA.py,
                                                        HDBSCAN CASA 2D.py)
              |                                                     |
              +------------------ compare with manual count --------+
```

| File | What it does |
|---|---|
| `FILTRATION OF TRACKMATE TRACKS.py` | Cleans TrackMate output: links track fragments that belong to one cell (gap, speed and distance limits), cuts tracks at the image border, detects collisions between cells and swaps wrongly exchanged identities, drops tracks shorter than 20 spots. All limits are set in the settings block at the top. Requires `networkx`. |
| `MANUAL CLUSTERIZATION MAIN.py` | Baseline: fixed speed and linearity thresholds taken from the WHO criteria assign RP / SP / NP / IM to every track. |
| `SPERM CLUSTERIZATION NO CASA.py` | Main method. For each video separately: clips outliers, applies a Yeo-Johnson transform, weights the features, runs HDBSCAN (the minimum cluster size is chosen from the number of tracks and retried if fewer than four clusters are found), names the clusters RP / SP / NP / IM by their position in feature space, assigns noise tracks to the nearest cluster and runs a sanity check. Writes labelled tables, a per-video summary and UMAP / scatter plots. |
| `HDBSCAN CASA 2D.py` | Newer variant that clusters CASA-style metrics (`CASA_VAP`, `CASA_STR`, optionally a third one) in two dimensions and merges the immotile clusters. Writes `tracks_labeled.xlsx`, `summary_all_videos.xlsx` and a scatter plot per video. |
| `Macros for spots.xlsm`, `Macros for tracks.xlsm` | Excel macros for processing the exported TrackMate spot and track tables. |

### Features used by `SPERM CLUSTERIZATION.py`

All come from the TrackMate track table:

- mean track speed (`TRACK_MEAN_SPEED`, px/frame);
- linearity of forward progression (`LINEARITY_OF_FORWARD_PROGRESSION`);
- log of the ratio between the maximum distance from the start and the net displacement (`MAX_DISTANCE_TRAVELED / TRACK_DISPLACEMENT`), which separates cells that swim away from cells that circle or wiggle in place.

HDBSCAN runs on these three scaled features. It does not run on the UMAP embedding.

## How to run

1. Track the cells in Fiji with TrackMate and export the spot and track tables to Excel.
2. Install the requirements:
   ```
   pip install numpy pandas scipy scikit-learn matplotlib openpyxl networkx hdbscan umap-learn
   ```
3. Open the script you need and edit the settings block at the top (input and output paths, thresholds, weights). The paths in the repository are examples from my own machine and have to be changed.
4. Run the scripts in the order of the diagram: filtering first, then clustering.

The sperm videos come from a human cohort and are **not included** in this repository. The scripts expect Excel tables with the TrackMate column names given above.

## Results

On 20 fresh ejaculate samples (video, tertiles by sperm concentration) the clustering was compared with the threshold classification and with the manual count of a qualified technician. HDBSCAN agreed best with the technician at low concentrations (first tertile: mean deviation 4.55%, maximum 9.0%); fixed thresholds were most accurate at high concentrations (third tertile: mean deviation 2.2%). This suggests a hybrid approach that depends on sample concentration. Details are in the poster.

## Limitations

- **No independent validation yet.** Feature weights, boundaries and post-processing parameters were chosen by hand while looking at the same 20 samples. The numbers above therefore describe how well the method fits these samples, not how well it will work on new ones. Validation on new samples and against a CASA system is the next step.
- **Fixed thresholds leak information.** An earlier version derived the thresholds from the average category shares of all videos and then applied them to every video. That uses the evaluation data to set the parameters and biases the result for particular videos. This is why the main method clusters each video separately.
- **Units depend on the setup.** Speed is in px/frame and the thresholds in the scripts were set for my camera and magnification. Recalibrate them for another microscope.
- **Small cohort, one imaging setup.**

## Authors and supervision

Dzhalil Salimov (Lomonosov Moscow State University, Faculty of Biotechnology). The track filtering, in particular cutting tracks at the image border and swapping collisions, was developed together with Timur Mamaev. Supervisors: Oleg Sergeyev (Skoltech) and Igor Kireev.
