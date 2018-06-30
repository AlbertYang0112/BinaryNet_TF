#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cmath>

using namespace tensorflow;

REGISTER_OP("ShiftBasedBatchNorm")
    .Input("to_norm: float")
    .Output("batch_normed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class ShiftBasedBatchNormOp : public OpKernel
{
    public:
        explicit ShiftBasedBatchNormOp(OpKernelConstruction* context): 
            OpKernel(context) {}
        void Compute(OpKernelContext* context) override;
    private:
        int ap2(float x);
};

void ShiftBasedBatchNormOp::Compute(OpKernelContext* context)
{

}

int ShiftBasedBatchNormOp::ap2(float x)
{
    // Todo: Access the error of the approximate function

    int shiftBits = int(log2f(abs(x)));
    return x >= 0 ? 1 << shiftBits : -1 << shiftBits;
}