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

var maxpred = flag.Int("maxpred", 3, "Max predicts without update")
var minupd = flag.Int("minupd", 2, "Min updates use prediction")
var ioutr = flag.Float64("ioutr", 0.4, "IOU threshold")

var sortCash = map[int64]*SORT{}

func main() {
	// logrus.SetLevel(logrus.DebugLevel)
	s := bufio.NewScanner(os.Stdin)
	bufsize := 10 << 20
	buf := make([]byte, bufsize)
	s.Buffer(buf, bufsize)
	for {
		if s.Scan() {
			reqdata := s.Bytes()
			// f, _ := os.Create("/home/hv8/data.txt")
			// defer f.Close()

			// _, err2 := f.WriteString(s.Text())
			// if err2 != nil {
			// 	log.Fatal(err2)
			// }
			cam_id := gjson.ParseBytes(reqdata).Get("camera.%did").Int()
			if sort, ok := sortCash[cam_id]; ok {
				track(reqdata, sort)
			} else {
				sort := NewSORT(*maxpred, *minupd, *ioutr)
				sortCash[cam_id] = &sort
				track(reqdata, &sort)
			}
		}

	}
}

func track(reqdata []byte, sort *SORT) {
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
	sort.Update(track_bboxes)
	for ind := range items.Array() {
		pathkey := fmt.Sprintf("items.%d.id", ind)
		reqdata, _ = sjson.SetBytes(reqdata, pathkey, strconv.FormatInt(sort.Trackers[ind].ID, 10))
	}

	fmt.Println(string(reqdata))

}
