"""
Prepare resnet model to deploy on Raspberry Pi. TVM compiles net on Mac and deploys
on Raspberry Pi at runtime. Uses RPC server.
https://docs.tvm.ai/tutorials/frontend/deploy_model_on_rasp.html

"""

import tvm
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime
from tvm.contrib.download import download_testdata

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def main():
    # one line to get the model
    block = get_model('resnet18_v1', pretrained=True)
    # test model
    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    img_name = 'cat.png'
    img_path = download_testdata(img_url, img_name, module='data')
    image = Image.open(img_path).resize((224, 224))
    # tvm specific data path
    # print(img_path)

    x = transform_image(image)

    # label number to word dict prepped with synset
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                          '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                          '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                          'imagenet1000_clsid_to_human.txt'])
    synset_name = 'imagenet1000_clsid_to_human.txt'
    synset_path = download_testdata(synset_url, synset_name, module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    print(synset)

    # Port GLuon model to portable computational graph
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape

    shape_dict = {'data': x.shape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    # we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)


if __name__ == "__main__":
    main()