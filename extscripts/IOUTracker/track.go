package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var maxlost = flag.Int("maxlost", 2, "Max lost detections")
var ioutr = flag.Float64("ioutr", 0.4, "IOU threshold")

var trackerCash = map[int64]*Tracker{}

func main() {
	// logrus.SetLevel(logrus.DebugLevel)
	s := bufio.NewScanner(os.Stdin)
	bufsize := 10 << 20
	buf := make([]byte, bufsize)
	s.Buffer(buf, bufsize)
	for {
		if s.Scan() {
			reqdata := s.Bytes()
			cam_id := gjson.ParseBytes(reqdata).Get("camera.%did").Int()
			if tracker, ok := trackerCash[cam_id]; ok {
				do_track(reqdata, tracker)
			} else {
				tracker := NewTracker(*maxlost, *ioutr)
				trackerCash[cam_id] = &tracker
				do_track(reqdata, &tracker)
			}
		}

	}
}

func do_track(reqdata []byte, tracker *Tracker) {
	items := gjson.ParseBytes(reqdata).Get("items")
	track_bboxes := [][]float64{}
	for _, item := range items.Array() {
		bbox := []float64{}
		item.Get("bbox").ForEach(func(key, value gjson.Result) bool {
			bbox = append(bbox, value.Num)
			return true
		})
		bbox = []float64{bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]}
		bbox_prob := append(bbox, item.Get("prob").Num)
		track_bboxes = append(track_bboxes, bbox_prob)
	}
	tracker.Update(track_bboxes)
	for ind := range items.Array() {
		pathkey := fmt.Sprintf("items.%d.id", ind)
		reqdata, _ = sjson.SetBytes(reqdata, pathkey, strconv.FormatInt(int64(tracker.Tracks[ind].ID), 10))
	}

	fmt.Println(string(reqdata))

}
