package main

type Tracker struct {
	maxLost      int
	iouThreshold float64
	FrameCount   int
	Tracks       []*Track
	nextId       int
}

type Track struct {
	Bbox     []float64
	Prob     float64
	ID       int
	Lost     int
	Age      int
	IOUScore float64
	FrameId  int
}

func MatchBboxes(curr_track *Track, dets [][]float64) (int, Track) {
	// TODO repair logic
	best_match := Track{}
	iou_curr := 0.0
	c_bbox := curr_track.Bbox
	best_ind := 0
	best_prob := curr_track.Prob
	for ind, bbox := range dets {
		prob := bbox[4]
		bbox = []float64{bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]}
		if IOU(curr_track.Bbox, bbox) > float64(iou_curr) {
			iou_curr = IOU(curr_track.Bbox, bbox)
			c_bbox = bbox
			best_ind = ind
			best_prob = prob

		}
	}
	best_match.Bbox = c_bbox
	best_match.Prob = best_prob
	best_match.ID = curr_track.ID
	return best_ind, best_match
}

func NewTracker(maxLost int, iouThreshold float64) Tracker {
	return Tracker{
		maxLost:      maxLost,
		iouThreshold: iouThreshold,
		FrameCount:   0,
		nextId:       1,
	}
}

func (s *Track) Update(frame int, newTrack Track, iouScore float64) {
	s.Bbox = newTrack.Bbox
	s.ID = newTrack.ID
	s.Prob = newTrack.Prob
	s.IOUScore = iouScore
	s.Age = s.Age + 1

}

func RemoveByIndex(s [][]float64, index int) [][]float64 {
	ret := make([][]float64, 0)
	ret = append(ret, s[:index]...)
	return append(ret, s[index+1:]...)
}

func (s *Tracker) RemoveTrackByIndexs(indxs []int) {
	var resultVec = []*Track{}

	for _, indx := range indxs {
		s.Tracks[indx] = nil
	}
	for _, val := range s.Tracks {
		if val != nil {
			resultVec = append(resultVec, val)
		}
	}
	s.Tracks = resultVec

}
func (s *Tracker) AddTrack(bbox []float64, prob float64) {
	new_track := Track{}
	new_track.Bbox = bbox
	new_track.Prob = prob
	new_track.ID = s.nextId
	new_track.FrameId = s.FrameCount
	s.nextId = s.nextId + 1
	s.Tracks = append(s.Tracks, &new_track)
}

func (s *Tracker) Update(dets [][]float64) {
	s.FrameCount = s.FrameCount + 1
	updates_track := []int{}
	deleted_tracks := []int{}
	for ind, track := range s.Tracks {
		if len(dets) > 0 {
			index, best_match := MatchBboxes(track, dets)
			iou := IOU(best_match.Bbox, track.Bbox)
			if iou >= s.iouThreshold {
				track.Update(s.FrameCount, best_match, iou)
				updates_track = append(updates_track, track.ID)
				RemoveByIndex(dets, index)
			}
		}
		if len(updates_track) == 0 || track.ID != updates_track[len(updates_track)-1] {
			track.Lost++
			if track.Lost > s.maxLost {
				deleted_tracks = append(deleted_tracks, ind)
			}

		}
	}
	if len(deleted_tracks) > 0 {
		s.RemoveTrackByIndexs(deleted_tracks)
	}

	for _, detect := range dets {
		bbox := []float64{detect[0], detect[1], detect[0] + detect[2], detect[1] + detect[3]}
		prob := detect[4]
		s.AddTrack(bbox, prob)
	}

}
