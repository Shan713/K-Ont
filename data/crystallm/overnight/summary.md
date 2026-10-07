# Phase C overnight summary

## phaseC_chgnet_large.json
```
{
 "composition": {
  "candidates_relaxed": 257,
  "materials": 52,
  "median_dE": 0.0001,
  "within_0.05_eV": 0.887,
  "within_0.10_eV": 0.918,
  "materials_best_within_0.05_eV": 51,
  "materials_best_below_true": 8,
  "match_after_relax": 171,
  "materials_match_after_relax": 42
 },
 "oracle": {
  "candidates_relaxed": 270,
  "materials": 54,
  "median_dE": 0.0,
  "within_0.05_eV": 0.941,
  "within_0.10_eV": 0.956,
  "materials_best_within_0.05_eV": 53,
  "materials_best_below_true": 6,
  "match_after_relax": 227,
  "materials_match_after_relax": 50
 }
}
```

## phaseC_chgnet_small_seeded.json
```
{
 "composition": {
  "candidates_relaxed": 259,
  "materials": 52,
  "median_dE": 0.001,
  "within_0.05_eV": 0.772,
  "within_0.10_eV": 0.815,
  "materials_best_within_0.05_eV": 50,
  "materials_best_below_true": 10,
  "match_after_relax": 126,
  "materials_match_after_relax": 36
 },
 "oracle": {
  "candidates_relaxed": 269,
  "materials": 54,
  "median_dE": 0.0005,
  "within_0.05_eV": 0.755,
  "within_0.10_eV": 0.814,
  "materials_best_within_0.05_eV": 48,
  "materials_best_below_true": 9,
  "match_after_relax": 162,
  "materials_match_after_relax": 43
 }
}
```

## phaseC_nearmiss_calib_large.json
```
{
 "composition | other": {
  "materials": 95,
  "valid_cifs": 950,
  "right_composition": 1.0,
  "right_space_group": 0.457,
  "strict_match": 0.46,
  "loose_match": 0.52,
  "materials_with_loose_match": 90,
  "materials_with_right_space_group": 84,
  "median_volume_ratio": 1.001
 },
 "oracle | other": {
  "materials": 95,
  "valid_cifs": 950,
  "right_composition": 1.0,
  "right_space_group": 0.998,
  "strict_match": 0.927,
  "loose_match": 0.934,
  "materials_with_loose_match": 93,
  "materials_with_right_space_group": 95,
  "median_volume_ratio": 1.001
 }
}
```

## phaseC_nearmiss_calib_small.json
```
{
 "composition | other": {
  "materials": 95,
  "valid_cifs": 949,
  "right_composition": 1.0,
  "right_space_group": 0.432,
  "strict_match": 0.421,
  "loose_match": 0.494,
  "materials_with_loose_match": 85,
  "materials_with_right_space_group": 81,
  "median_volume_ratio": 0.997
 },
 "oracle | other": {
  "materials": 95,
  "valid_cifs": 948,
  "right_composition": 1.0,
  "right_space_group": 0.991,
  "strict_match": 0.914,
  "loose_match": 0.923,
  "materials_with_loose_match": 94,
  "materials_with_right_space_group": 95,
  "median_volume_ratio": 1.0
 }
}
```

## phaseC_nearmiss_large.json
```
{
 "composition | cathode-like": {
  "materials": 55,
  "valid_cifs": 501,
  "right_composition": 0.996,
  "right_space_group": 0.703,
  "strict_match": 0.635,
  "loose_match": 0.649,
  "materials_with_loose_match": 46,
  "materials_with_right_space_group": 47,
  "median_volume_ratio": 1.0
 },
 "oracle | cathode-like": {
  "materials": 55,
  "valid_cifs": 539,
  "right_composition": 0.998,
  "right_space_group": 0.996,
  "strict_match": 0.814,
  "loose_match": 0.855,
  "materials_with_loose_match": 50,
  "materials_with_right_space_group": 54,
  "median_volume_ratio": 1.0
 },
 "rf_top5 | cathode-like": {
  "materials": 55,
  "valid_cifs": 480,
  "right_composition": 0.642,
  "right_space_group": 0.11,
  "strict_match": 0.11,
  "loose_match": 0.121,
  "materials_with_loose_match": 29,
  "materials_with_right_space_group": 26,
  "median_volume_ratio": 0.99
 }
}
```

## phaseC_nearmiss_sanity_large.json
```
{
 "composition | cathode-like": {
  "materials": 20,
  "valid_cifs": 179,
  "right_composition": 0.994,
  "right_space_group": 0.162,
  "strict_match": 0.145,
  "loose_match": 0.151,
  "materials_with_loose_match": 5,
  "materials_with_right_space_group": 5,
  "median_volume_ratio": 1.035
 },
 "oracle | cathode-like": {
  "materials": 20,
  "valid_cifs": 172,
  "right_composition": 0.715,
  "right_space_group": 0.994,
  "strict_match": 0.291,
  "loose_match": 0.308,
  "materials_with_loose_match": 7,
  "materials_with_right_space_group": 19,
  "median_volume_ratio": 0.96
 }
}
```

## phaseC_nearmiss_sanity_small.json
```
{
 "composition | cathode-like": {
  "materials": 20,
  "valid_cifs": 189,
  "right_composition": 0.974,
  "right_space_group": 0.206,
  "strict_match": 0.196,
  "loose_match": 0.212,
  "materials_with_loose_match": 7,
  "materials_with_right_space_group": 7,
  "median_volume_ratio": 1.03
 },
 "oracle | cathode-like": {
  "materials": 20,
  "valid_cifs": 196,
  "right_composition": 0.827,
  "right_space_group": 0.969,
  "strict_match": 0.372,
  "loose_match": 0.444,
  "materials_with_loose_match": 14,
  "materials_with_right_space_group": 20,
  "median_volume_ratio": 0.962
 }
}
```

## phaseC_nearmiss_small_seeded.json
```
{
 "composition | cathode-like": {
  "materials": 55,
  "valid_cifs": 503,
  "right_composition": 0.978,
  "right_space_group": 0.64,
  "strict_match": 0.467,
  "loose_match": 0.503,
  "materials_with_loose_match": 41,
  "materials_with_right_space_group": 48,
  "median_volume_ratio": 0.997
 },
 "oracle | cathode-like": {
  "materials": 55,
  "valid_cifs": 540,
  "right_composition": 0.974,
  "right_space_group": 0.991,
  "strict_match": 0.581,
  "loose_match": 0.626,
  "materials_with_loose_match": 43,
  "materials_with_right_space_group": 54,
  "median_volume_ratio": 1.0
 },
 "rf_top5 | cathode-like": {
  "materials": 55,
  "valid_cifs": 474,
  "right_composition": 0.646,
  "right_space_group": 0.105,
  "strict_match": 0.078,
  "loose_match": 0.086,
  "materials_with_loose_match": 21,
  "materials_with_right_space_group": 26,
  "median_volume_ratio": 0.998
 }
}
```
