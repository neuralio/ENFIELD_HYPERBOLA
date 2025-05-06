1. In `config` file you can change the parameters (paths to images, neural network params etc)
   
   * In  `MODEL__CLASSES` you can  select which  models to train.
   
   * Inference uses the same list , so you can select - unselect on which models you want to apply inference.

2. ###### Training
   
   * You run the `main` script.

3. ###### Inference
   
   * You run the `inference` script.

4. ###### Inference on edge devices (convert to tensor RT)
   
   a. Run `export_to_onnx`
   
   b. Run `onnx_to_tensorrt`
   
   c. Run `edge_inference`

5. ###### Benchmark
   
   You can run `benchmark` script to benchmark `inference` and `edge_inference`
   
   Right now uses only `Unet` model (`model_path` , `engine_path`,`model = HybridModel_3`, `model_name="Unet"`).


