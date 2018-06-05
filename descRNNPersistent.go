package gocudnn

/*
#include <cudnn.h>
*/
import "C"

//PersistentRNNPlan holds  C.cudnnPersistentRNNPlan_t
type PersistentRNNPlan struct {
	plan C.cudnnPersistentRNNPlan_t
}

//CreatePersistentRNNPlan creates a PersistentRNNPlan
func (r *RNND) CreatePersistentRNNPlan(minibatch int32, data DataType) (PersistentRNNPlan, error) {
	var plan C.cudnnPersistentRNNPlan_t
	err := Status(C.cudnnCreatePersistentRNNPlan(
		r.descriptor,
		C.int(minibatch),
		data.c(),
		&plan,
	)).error("CreatePersistentRNNPlan")
	return PersistentRNNPlan{
		plan: plan}, err
}

//SetPersistentRNNPlan sets a SetPersistentRNNPlan
func (r *RNND) SetPersistentRNNPlan(plan PersistentRNNPlan) error {
	return Status(C.cudnnSetPersistentRNNPlan(r.descriptor, plan.plan)).error("SetPersistentRNNPlan")
}

//DestroyPersistentRNNPlan destroys the C.cudnnPersistentRNNPlan_t in the PersistentRNNPlan struct
func (p *PersistentRNNPlan) DestroyPersistentRNNPlan() error {
	return Status(C.cudnnDestroyPersistentRNNPlan(p.plan)).error("DestroyPersistentRNNPlan")
}
