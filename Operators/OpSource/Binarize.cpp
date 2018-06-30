#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Binarize")
    .Input("to_binary: float")
    .Output("binarized: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
    {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class BinarizeOp : public OpKernel
{
    public:
        explicit BinarizeOp(OpKernelConstruction* context):
            OpKernel(context) {}
        void Compute(OpKernelContext* context) override;
};

void BinarizeOp::Compute(OpKernelContext* context)
{

    // Todo: Compress the output tensor
    // Todo: Add the stochastic binarization method

    // Grab the input tensor
    const Tensor& inputTensor = context->input(0);
    auto input = inputTensor.flat<float>();
    
    // Create an output tensor
    Tensor* binarizedTensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, inputTensor.shape(), &binarizedTensor)
    );
    auto binarizedFlat = binarizedTensor->flat<int32>();

    const int N = input.size();
    for(int i = 0; i < N; i++)
    {
        // Deterministic binarization
        binarizedFlat(i) = input(i) >= 0 ? 1 : -1;
    }
}

REGISTER_KERNEL_BUILDER(Name("Binarize").Device(DEVICE_CPU), BinarizeOp);