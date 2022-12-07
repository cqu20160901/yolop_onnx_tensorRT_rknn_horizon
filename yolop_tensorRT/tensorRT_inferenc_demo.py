import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from math import exp
from math import sqrt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


CLASSES = ['car']
class_num = len(CLASSES)

anchor_num = 3
output_head = 3
cell_size = [[48, 80], [24, 40], [12, 20]]
anchor_size = [
    [[3, 9], [5, 11], [4, 20]],
    [[7, 18], [6, 39], [12, 31]],
    [[19, 50], [38, 81], [68, 157]]]


stride = [8, 16, 32]
grid_cell = np.zeros(shape=(output_head, 48, 80, 2))

nms_thre = 0.45
obj_thre = [0.4]

input_imgW = 640
input_imgH = 384


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine_from_bin(engine_file_path):
    print('Reading engine from file {}'.format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]



class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, head):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.head = head


def grid_cell_init():
    for index in range(output_head):
        for w in range(cell_size[index][1]):
            for h in range(cell_size[index][0]):
                grid_cell[index][h][w][0] = w
                grid_cell[index][h][w][1] = h


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nms_thre:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def det_postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []

    output = []
    for i in range(len(out)):
        output.append(out[i].reshape((-1)))

    gs = 4 + 1 + class_num
    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    for head in range(output_head):
        y = output[head]
        for h in range(cell_size[head][0]):

            for w in range(cell_size[head][1]):
                for a in range(anchor_num):
                    conf_scale = sigmoid(y[((a * gs + 4) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w])
                    for cl in range(class_num):
                        conf = sigmoid(y[((a * gs + 5 + cl) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w]) * conf_scale

                        if conf > obj_thre[cl]:
                            bx = (sigmoid(y[((a * gs + 0) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w]) * 2.0 - 0.5 + grid_cell[head][h][w][0]) * stride[head]
                            by = (sigmoid(y[((a * gs + 1) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w]) * 2.0 - 0.5 + grid_cell[head][h][w][1]) * stride[head]
                            bw = pow((sigmoid(y[((a * gs + 2) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w]) * 2), 2) * anchor_size[head][a][0]
                            bh = pow((sigmoid(y[((a * gs + 3) * cell_size[head][0] * cell_size[head][1]) + h * cell_size[head][1] + w]) * 2), 2) * anchor_size[head][a][1]

                            xmin = (bx - bw / 2) * scale_w
                            ymin = (by - bh / 2) * scale_h
                            xmax = (bx + bw / 2) * scale_w
                            ymax = (by + bh / 2) * scale_h

                            xmin = xmin if xmin > 0 else 0
                            ymin = ymin if ymin > 0 else 0
                            xmax = xmax if xmax < img_w else img_w
                            ymax = ymax if ymax < img_h else img_h

                            if xmin >= 0 and ymin >= 0 and xmax <= img_w and ymax <= img_h:
                                box = DetectBox(cl, conf, xmin, ymin, xmax, ymax, head)
                                detectResult.append(box)

    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    return predBox


def da_seg_postprocess(out, img_h, img_w):
    print('da_seg_postprocess ... ')
    output = out[3].reshape((-1))

    mask = np.zeros(shape=(input_imgH, input_imgW, 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if output[i * input_imgW + j] < output[input_imgH * input_imgW + i * input_imgW + j]:
                mask[i, j, 0] = 1
                mask[i, j, 1] = 1
                mask[i, j, 2] = 1
            else:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 0

    mask = mask * np.array([[[0, 0, 255]]])
    mask = cv2.resize(mask, (img_w, img_h))
    mask = mask.astype("uint8")

    return mask


def ll_seg_postprocess(out, img_h, img_w):
    print('ll_seg_postprocess ... ')

    output = out[4].reshape((-1))

    mask = np.zeros(shape=(input_imgH, input_imgW, 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if output[i * input_imgW + j] < output[input_imgH * input_imgW + i * input_imgW + j]:
                mask[i, j, 0] = 1
                mask[i, j, 1] = 1
                mask[i, j, 2] = 1
            else:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 0

    mask = mask * np.array([[[255, 0, 0]]])
    mask = cv2.resize(mask, (img_w, img_h))
    mask = mask.astype("uint8")

    return mask


def preprocess(src):
    img = cv2.resize(src, (input_imgW, input_imgH))
    img = img.astype(np.float32)
    img = img / 255

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img -= mean
    img /= std

    img = img.transpose(2, 0, 1)
    img_input = img.copy()
    return img_input


def main():
    engine_file_path = 'yolop_640x384.trt'
    input_image_path = 'test.jpg'

    origimg1 = cv2.imread(input_image_path)
    img_h, img_w = origimg1 .shape[:2]

    orig = cv2.cvtColor(origimg1 , cv2.COLOR_BGR2RGB)
    image = preprocess(orig)
    
    with get_engine_from_bin(engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        inputs[0].host = image
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        print(len(trt_outputs))

        out = []
        for i in range(len(trt_outputs)):
            print(len(trt_outputs[i]))
            out.append(trt_outputs[i])

        predbox = det_postprocess(out, img_h, img_w)
        da_seg_mask = da_seg_postprocess(out, img_h, img_w)
        ll_seg_mask = ll_seg_postprocess(out, img_h, img_w)

        for i in range(len(predbox)):
            xmin = int(predbox[i].xmin)
            ymin = int(predbox[i].ymin)
            xmax = int(predbox[i].xmax)
            ymax = int(predbox[i].ymax)
            classId = predbox[i].classId
            score = predbox[i].score
            head = predbox[i].head

            cv2.rectangle(origimg1, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            ptext = (xmin, ymin)
            title = CLASSES[classId] + "%.2f" % score
            cv2.putText(origimg1, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        origimg1 = np.clip(np.array(origimg1) + np.array(da_seg_mask) * 0.5, a_min=0, a_max=255)
        origimg1 = np.clip(np.array(origimg1) + np.array(ll_seg_mask) * 0.6, a_min=0, a_max=255)
        cv2.imwrite('./tensorrt_result.jpg', origimg1)


if __name__ == '__main__':
    print('This is main ...')
    grid_cell_init()
    main()
