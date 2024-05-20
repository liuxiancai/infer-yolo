- Currently supports Yolo series 3/4/5/x/7/8

### How to use the infer-yolo

### step1 compile the cmakelist.txt

### step2 Compile the model, e.g.
`trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine`

### step3: Use infer inference
`main函数中有三个子函数，包括perf(动态Batch)、batch_inference(静态batch)、single_inference(batch_size为1)`

# Reference

- [🌻shouxieai](https://github.com/shouxieai)

