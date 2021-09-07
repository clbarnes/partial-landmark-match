# partial-landmark-match

## Problem

- There are 2x3D spaces, X and Y, each with some locations marked
- Locations have a label, known or unknown, unique within their space
- The labels of locations in X are known
- The locations in Y match with a subset of locations in X
- The labels of locations in Y are largely unknown

## Suggested algorithm

- Use matched locations (i.e. exist in X and Y and share a label) as control points in moving least squares
- Find the closest location in Y (`L`) to the control points
- Transform `L` to X-space, and find its closest unpaired location
- Take the label from that partner
- Weight the match by comparing the distance^2 to the closest 2 prospective matches
- Add the pair as a control point, with that weighting

This is greedy, so it could go off the rails quite easily, but will hopefully grow out from understood regions to not-understood.
