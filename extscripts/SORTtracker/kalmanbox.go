package main

import (
	"fmt"

	"github.com/flaviostutz/kalman"
	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

var (
	lastID = int64(0)
)

//KalmanBoxTracker   This class represents the internel state of individual tracked objects observed as bbox.
type KalmanBoxTracker struct {
	ID                    int64
	Updates               int
	Predicts              int
	PredictsSinceUpdate   int
	UpdatesWithoutPredict int
	SkipPredicts          int
	LastBBox              []float64
	LastBBoxIOU           []float64
	// history               [][]float64
	LastResiduals []float64
	KalmanFilter  kalman.Filter
	KalmanCtrl    *mat.VecDense
	KalmanCtx     *kalman.Context
}

//NewKalmanBoxTracker     Initialises a tracker using initial bounding box.
func NewKalmanBoxTracker(bbox []float64) (KalmanBoxTracker, error) {
	if len(bbox) < 4 {
		return KalmanBoxTracker{}, fmt.Errorf("bbox should contain at least 4 positions: x1,y1,x2,y2")
	}
	//define constant velocity model
	kf := kalman.NewFilter(
		lti.Discrete{
			Ad: mat.NewDense(7, 7, []float64{
				1, 0, 0, 0, 1, 0, 0,
				0, 1, 0, 0, 0, 1, 0,
				0, 0, 1, 0, 0, 0, 1,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0,
				0, 0, 0, 0, 0, 1, 0,
				0, 0, 0, 0, 0, 0, 1}),
			Bd: mat.NewDense(7, 7, nil),
			C: mat.NewDense(4, 7, []float64{
				1, 0, 0, 0, 0, 0, 0,
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0}),
			D: mat.NewDense(4, 7, nil),
		},
		kalman.Noise{
			Q: mat.NewDense(7, 7, []float64{
				1, 0, 0, 0, 0, 0, 0,
				0, 1, 0, 0, 0, 0, 0,
				0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 1, 0, 0, 0,
				0, 0, 0, 0, 0.01, 0, 0,
				0, 0, 0, 0, 0, 0.01, 0,
				0, 0, 0, 0, 0, 0, 0.0001}),
			R: mat.NewDense(4, 4, []float64{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 10, 0,
				0, 0, 0, 10}),
		},
	)

	kctx := kalman.Context{
		X: mat.NewVecDense(7, []float64{0, 0, 0, 0, 0, 0, 0}),
		P: mat.NewDense(7, 7, []float64{
			10, 0, 0, 0, 1, 0, 0,
			0, 10, 0, 0, 0, 1, 0,
			0, 0, 10, 0, 0, 0, 1,
			0, 0, 0, 10, 0, 0, 0,
			0, 0, 0, 0, 1000, 0, 0,
			0, 0, 0, 0, 0, 10, 0,
			0, 0, 0, 0, 0, 0, 10}),
	}
	// self.M = np.zeros((dim_z, dim_z)) # process-measurement cross correlation
	// self.K = np.zeros((dim_x, dim_z)) # kalman gain
	// self.S = np.zeros((dim_z, dim_z)) # system uncertainty
	// self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

	ctrl := mat.NewVecDense(7, nil)

	z := mat.NewVecDense(4, convertBBoxToZ(bbox))
	kf.Apply(&kctx, z, ctrl)

	lastID = lastID + 1

	kbt := KalmanBoxTracker{
		ID:                    lastID,
		Updates:               0,
		UpdatesWithoutPredict: 0,
		Predicts:              0,
		PredictsSinceUpdate:   0,
		LastBBox:              bbox,
		KalmanFilter:          kf,
		KalmanCtrl:            ctrl,
		KalmanCtx:             &kctx,
		LastResiduals:         []float64{-1, -1, -1, -1},
		// history:               [][]float64{},
	}

	// kbt.Update(bbox)

	return kbt, nil
}

//Update     Updates the state vector with observed bbox
//Returns the residuals that is the difference between the real value (bbox) and the predicted value
func (k *KalmanBoxTracker) Update(bbox []float64) ([]float64, error) {
	if len(bbox) < 4 {
		return []float64{}, fmt.Errorf("bbox should contain at least 4 positions: x1,y1,x2,y2")
	}
	k.PredictsSinceUpdate = 0
	// k.history = [][]float64{}
	k.Updates = k.Updates + 1
	k.UpdatesWithoutPredict = k.UpdatesWithoutPredict + 1
	k.LastBBox = bbox

	cpred := k.CurrentPrediction()
	residuals := []float64{bbox[0] - cpred[0], bbox[1] - cpred[1], bbox[2] - cpred[2], bbox[3] - cpred[3]}

	z := mat.NewVecDense(4, convertBBoxToZ(bbox))

	k.KalmanFilter.Apply(k.KalmanCtx, z, k.KalmanCtrl)

	return residuals, nil
}

//PredictNext     Advances the state vector and returns the predicted bounding box estimate.
func (k *KalmanBoxTracker) PredictNext() []float64 {
	k.SkipPredicts = 0
	x := k.KalmanCtx.X
	if x.AtVec(6)+x.AtVec(2) <= 0 {
		x.SetVec(6, 0.0)
	}
	k.Predicts = k.Predicts + 1
	if k.PredictsSinceUpdate > 0 {
		k.UpdatesWithoutPredict = 0
	}

	//use auto prediction made during "Apply()" for the first predict request
	state := x
	if k.PredictsSinceUpdate > 0 {
		state = k.KalmanFilter.PredictState(k.KalmanCtx, k.KalmanCtrl)
	}
	k.PredictsSinceUpdate = k.PredictsSinceUpdate + 1

	z := []float64{state.AtVec(0), state.AtVec(1), state.AtVec(2), state.AtVec(3)}
	// k.history = append(k.history, bbox)
	return convertZToBBox(z)
}

//CurrentState Returns the current bounding box estimate.
func (k *KalmanBoxTracker) CurrentState() []float64 {
	state := k.KalmanFilter.CurrentState()
	z := []float64{state.AtVec(0), state.AtVec(1), state.AtVec(2), state.AtVec(3)}
	return convertZToBBox(z)
}

//CurrentPrediction get last prediction results
func (k *KalmanBoxTracker) CurrentPrediction() []float64 {
	k.SkipPredicts = 0
	state := k.KalmanCtx.X
	z := []float64{state.AtVec(0), state.AtVec(1), state.AtVec(2), state.AtVec(3)}
	return convertZToBBox(z)
}

// filter := kalman.NewFilter(
// 	X, // initial state (n x 1)
// 	P, // initial process covariance (n x n)
// 	F, // prediction matrix (n x n)
// 	B, // control matrix (n x k)
// 	Q, // process model covariance matrix (n x n)
// 	H, // measurement matrix (l x n)
// 	R, // measurement errors (l x l)
// )

// Ad - F
// Bd - B
// X - X
// P - P
// Q - Q
// C - H
// R - R

// X, // initial state (n x 1)
// P, // initial process covariance (n x n)
// Ad, // prediction matrix (n x n)
// Bd, // control matrix (n x k)
// Q, // process model covariance matrix (n x n)
// C,  // measurement matrix (l x n)
// R, // measurement errors (l x l)
// D,  // measurement matrix (l x k)