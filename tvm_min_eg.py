"""
Minimum example TVM.
https://docs.tvm.ai/tutorials/language/extern_op.html#sphx-glr-download-tutorials-language-extern-op-py

"""

from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import cblas

def main():
    ctx = tvm.cpu(0)
    n = 1024
    l = 128
    m = 235
    bias = tvm.var('bias', dtype=tvm.float32)
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((l, m), name='B')
    C = tvm.extern((n, m), [A, B], lambda ins, outs: tvm.call_packed("tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], False, False), name="C")
    D = tvm.compute(C.shape, lambda i, j: C(i, j) + bias, name="D")
    s = tvm.create_schedule(D.op)
    f = tvm.build(s, [A, B, D, bias], "llvm")
    a = tvm.nd.array(np.random.uniform(size=(n,l)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
    d = tvm.nd.array(np.zeros((n, m), dtype = D.dtype), ctx)
    bb = 10.0
    print(d.asnumpy())
    tvm.testing.assert_allclose(d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + 10, rtol=1e-5)

if __name__ == "__main__":
    main()
